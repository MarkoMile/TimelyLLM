import asyncio
import time

from transformers import PreTrainedTokenizer
from typing import (AsyncIterator, Callable, Dict, Iterable, List, Mapping,
                    Optional, Set, Tuple, Type, Union)
from vllm.engine.async_timeout import asyncio_timeout
from vllm.logger import init_logger
import vllm.envs as envs
from vllm.engine.llm_engine import LLMEngine


logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S

class AsyncEngineDeadError(RuntimeError):
    pass

# add stop condition
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import Sequence, SequenceStatus
from vllm.sampling_params import SamplingParams
from rtengine.stop_rule import MiniSpecProgram
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
# model_path = "/home/neiwen/hf_modelzoo/models--meta-llama--Meta-Llama-3-8B-instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
# model_path = "/home/neiwen/hfmodelzoo/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
# model_path = "/home/neiwen/hfmodelzoo/models--microsoft--Phi-3-mini-128k-instruct/snapshots/a90b62ae09941edff87a90ced39ba5807e6b2ade"



class CustomStopChecker(StopChecker):
    def __init__(self, max_model_len: int, get_tokenizer_for_seq: Callable[[Sequence], PreTrainedTokenizer]):
        super().__init__(max_model_len, get_tokenizer_for_seq)
        self.program_interpreter = MiniSpecProgram()

    def maybe_stop_sequence(self, seq: Sequence, new_char_count: int, sampling_params: SamplingParams, lora_req=None) -> None:
        # super().maybe_stop_sequence(seq, new_char_count, sampling_params, lora_req)
        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return
        
        # Add custom stop condition logic here
        # for typefly
        flag_executable = self.program_interpreter.parse(seq.output_text, True)
        if flag_executable:
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = "stop by checker"
            return


class CustomStopChecker4FLTRNN(StopChecker):
    def __init__(self, max_model_len: int, get_tokenizer_for_seq: Callable[[Sequence], PreTrainedTokenizer]):
        super().__init__(max_model_len, get_tokenizer_for_seq)
        self.program_interpreter = MiniSpecProgram()

    def maybe_stop_sequence(self, seq: Sequence, new_char_count: int, sampling_params: SamplingParams, lora_req=None) -> None:
        # super().maybe_stop_sequence(seq, new_char_count, sampling_params, lora_req)
        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return
        
        # Add custom stop condition logic here
        # for FLTRNN
        if ';' in seq.output_text:
            flag_executable =  True
        else: 
            flag_executable = False
        if flag_executable:
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = "stop by checker"
            return
        

class CustomStopChecker4Chatbot(StopChecker):
    def __init__(self, max_model_len: int, get_tokenizer_for_seq: Callable[[Sequence], PreTrainedTokenizer]):
        super().__init__(max_model_len, get_tokenizer_for_seq)

    def maybe_stop_sequence(self, seq: Sequence, new_char_count: int, sampling_params: SamplingParams, lora_req=None) -> None:
        # super().maybe_stop_sequence(seq, new_char_count, sampling_params, lora_req)
        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return
        
        # Add custom stop condition logic here
        LENGTH_THRESHOLD = 10
        if any(punct in seq.output_text for punct in ['.', '!', '?']):
            if seq.get_output_len() > LENGTH_THRESHOLD:
                flag_executable = True
            else:
                flag_executable = False
        else:
            flag_executable = False
        if flag_executable:
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = "stop by checker"
            return

class RTLLMEngine(LLMEngine):
    def initialize_stop_checker(self, robot_system: str = "typefly"):
        # stop_checker = CustomStopChecker(
        #     max_model_len=1024, # need to adjust it according to the max_token in SamplingParams
        #     get_tokenizer_for_seq=lambda seq: PreTrainedTokenizer.from_pretrained(model_path)
        # )    
        if robot_system == "typefly":
            stop_checker = CustomStopChecker(
                max_model_len=1024, # need to adjust it according to the max_token in SamplingParams
                get_tokenizer_for_seq=lambda seq: PreTrainedTokenizer.from_pretrained(model_path)
            )
        elif robot_system == "fltrnn":
            stop_checker = CustomStopChecker4FLTRNN(
                max_model_len=1024, # need to adjust it according to the max_token in SamplingParams
                get_tokenizer_for_seq=lambda seq: PreTrainedTokenizer.from_pretrained(model_path)
            )
        elif robot_system == "chatbot":
            stop_checker = CustomStopChecker4Chatbot(
                max_model_len=1024, # need to adjust it according to the max_token in SamplingParams
                get_tokenizer_for_seq=lambda seq: PreTrainedTokenizer.from_pretrained(model_path)
            )
        else:
            raise ValueError(f"Unsupported robot system: {robot_system}")
        # if self.engine_use_ray:
        #     self._initialize_stop_checker.remote(stop_rule=stop_checker)
        # else:
        self._initialize_stop_checker(stop_rule=stop_checker)

    
    def _initialize_stop_checker(self, stop_rule:StopChecker)-> None:
        # self.stop_checker=stop_rule
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                self.get_tokenizer_for_seq,
                stop_rule
            ))