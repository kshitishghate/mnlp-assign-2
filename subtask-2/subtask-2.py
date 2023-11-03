from datasets import load_dataset, load_metric, Audio, ClassLabel
import re
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import argparse
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer



#Login to huggingface and get the token

def preprocess_data(args):
    common_voice_train = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split="train+validation",use_auth_token=True)
    common_voice_test = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split="test",use_auth_token=True)
    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    if args.model_name != "whisper-small":

        # Remove meaningless characters
        chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

        def remove_special_characters(batch):
            batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
            return batch

        common_voice_train = common_voice_train.map(remove_special_characters)
        common_voice_test = common_voice_test.map(remove_special_characters)

        #Create Vocab List:
        def extract_all_chars(batch):
            all_text = " ".join(batch["sentence"])
            vocab = list(set(all_text))
            return {"vocab": [vocab], "all_text": [all_text]}

        vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
        vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
        vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        #Save vocab file
        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        #Tokenizer:

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

        #Feature Extractor:
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        #Processor
        if args.model_name== "XLS-R-spanish":
            processor = Wav2Vec2Processor.from_pretrained(args.pretrained_checkpoint)
        else:
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        

    else:
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="spanish", task="transcribe")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")


    def prepare_dataset(batch):
        audio = batch["audio"]

        if args.model_name != "whisper-small":
            # batched output is "un-batched"
            batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            batch["input_length"] = len(batch["input_values"])

            with processor.as_target_processor():
                batch["labels"] = processor(batch["sentence"]).input_ids
        else:
            # compute log-Mel input features from input audio array 
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids 
            batch["labels"] = tokenizer(batch["sentence"]).input_ids

        return batch
    
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)

    return common_voice_train, common_voice_test, processor, tokenizer


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
def train(args, common_voice_train,common_voice_test, processor, tokenizer):
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        if args.model_name != "whisper-small":
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        else:
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)


        wer = 100*wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    if args.model_name != "whisper-small":

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        model = Wav2Vec2ForCTC.from_pretrained(
                args.pretrained_checkpoint,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                feat_proj_dropout=0.0,
                mask_time_prob=0.05,
                layerdrop=0.0,
                ctc_loss_reduction="mean",
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer),
            )
        
        model.freeze_feature_extractor()

        training_args = TrainingArguments(
                        output_dir=args.repo_name,
                        group_by_length=True,
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=2,
                        evaluation_strategy="steps",
                        num_train_epochs=30,
                        gradient_checkpointing=True,
                        fp16=True,
                        save_steps=400,
                        eval_steps=400,
                        logging_steps=400,
                        learning_rate=3e-4,
                        warmup_steps=500,
                        save_total_limit=2,
                        push_to_hub=True,
                        )
        trainer = Trainer(
                            model=model,
                            data_collator=data_collator,
                            args=training_args,
                            compute_metrics=compute_metrics,
                            train_dataset=common_voice_train,
                            eval_dataset=common_voice_test,
                            tokenizer=processor.feature_extractor,
                        )
        
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        model = WhisperForConditionalGeneration.from_pretrained(args.pretrained_checkpoint)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []   

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.repo_name,  # change to a repo name of your choice
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=True,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=common_voice_train,
            eval_dataset=common_voice_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        
    
    trainer.train()

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="XLS-R",choices=('XLS-R', 'XLS-R-spanish', 'whisper-small'))
    parser.add_argument("--repo_name", type=str)
    args = parser.parse_args()

    if args.model_name == "XLS-R":
        args.pretrained_checkpoint = "facebook/wav2vec2-xls-r-300m"
    elif args.model_name == "XLS-R-spanish":
        args.pretrained_checkpoint = "facebook/wav2vec2-large-xlsr-53-spanish"
    else:
        args.pretrained_checkpoint = "openai/whisper-small"

    print("Preparing data")
    common_voice_train, common_voice_test, processor, tokenizer = preprocess_data(args)

    print("Fine-tuning model")
    train(args, common_voice_train,common_voice_test, processor, tokenizer)
   
