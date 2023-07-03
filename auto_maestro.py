"""

Bot that generates

"""
from __future__ import annotations

from typing import AsyncIterable, Optional, Any

from fastapi_poe import PoeBot, run
from fastapi_poe.types import QueryRequest, ProtocolMessage
from fastapi_poe.client import MetaMessage, stream_request,get_final_response
from sse_starlette.sse import ServerSentEvent

import replicate
import asyncio

from urllib.parse import urlparse, quote
from dataclasses import dataclass
import re

_WAIT_TIMEOUT_S = 1


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


melody_prompt_pattern = (
    r"^(?:melody:\s*(?P<melody>.*?)\s*)?(?:prompt:\s*(?P<prompt>.*?)\s*)?$"
)

previous_prompt_pattern = (
    r"(?<=Generated from prompt: )(.*?)(?=\n\nReference melody)"
)

def _parse_generated_audio_url_from_message(message: ProtocolMessage) -> Optional[str]:
    pattern = r"\[(?:.*?)\]\((.*?)\)"
    match = re.search(pattern, message.content)

    if match:
        generated_audio_url = match.group(1)
        return generated_audio_url
    else:
        return None

def _parse_previous_prompt_from_message(message: ProtocolMessage) -> Optional[str]:
    match = re.search(previous_prompt_pattern, message.content)

    if match:
        previous_prompt = match.group(1)
        return previous_prompt
    else:
        return None

def _get_last_generation_message(messages:list[ProtocolMessage])->Optional[ProtocolMessage]:
    for message in reversed(messages):
        if message.content.lstrip().startswith("Completed"):
            return message



_MESSAGE_FORMAT = """
Message format:
```
melody: <melody URL>
prompt: <a description of how the music should sound>
```
"""


def _linkify(phrase: str, prompt: str) -> str:
    return (
        f"[{phrase}](poe://www.poe.com/_api/key_phrase"
        f"?phrase={quote(phrase)}"
        f"&prompt={quote(prompt)})"
    )


def _find_previous_melody(messages: list[ProtocolMessage]) -> Optional[str]:
    for message in reversed(messages):
        if user_input := _parse_user_input(message):
            if user_input.melody_url is not None:
                return user_input.melody_url
    return None


def _find_previous_prompt(messages: list[ProtocolMessage]) -> Optional[str]:
    for message in reversed(messages):
        if user_input := _parse_user_input(message):
            if user_input.prompt is not None:
                return user_input.prompt
    return None


@dataclass
class UserInput:
    melody_url: Optional[str]
    prompt: Optional[str]


def _parse_user_input(message: ProtocolMessage) -> Optional[UserInput]:
    match = re.search(melody_prompt_pattern, message.content)
    if not match:
        return None

    melody_url = match.group("melody")
    prompt = match.group("prompt")
    return UserInput(melody_url=melody_url, prompt=prompt)


def _gen_melody_list(melody_urls: list[str]) -> str:
    melody_list = []
    for i, melody_url in enumerate(melody_urls):
        melody_list.append(
            f"[Melody {i+1}]({melody_url}) - {_linkify('<Select>', 'melody: '+ melody_url)}"
        )

    return "\n".join(melody_list)


def _gen_prompt_list(prompt_title_to_prompt: dict[str, str]) -> str:
    prompt_list = []
    for title, prompt in prompt_title_to_prompt.items():
        prompt_list.append(f"{_linkify(title, 'prompt: ' + prompt)}")

    return "\n".join(prompt_list)


_INTRO_MELODIES = [
    "https://dl.sndup.net/fkxt/San%20holo.m4a",
    "https://dl.sndup.net/k6kv/Porter.m4a",
    "https://dl.sndup.net/z7kk/New%20Recording%2074.m4a",
    "https://dl.sndup.net/z2g8/Porter%20Robinson%20-%20Sad%20Machine.m4a"
]
_SKIP_MESSAGE = "<SKIP>"

_WELCOME_MESSAGE = f"""Welcome to AutoMaestro's studio!

Describe to me the music you want to hear and (optionally) a source melody, and I'll write you a banger within seconds.

First things first, you can provide a URL to your source melody, or click {_linkify("<Skip>", _SKIP_MESSAGE)} if you only want to use a prompt.

To get a URL for your melody file, you can use https://sndup.net/ to upload your file and click "Download audio!" to get the file URL.

To use your own melody, say:

```
melody: <URL to your melody>
```

Alternatively, you can use one of these pre-recorded options. Click on the URL to listen to the melody and click "<Select>" to choose that melody.

{_gen_melody_list(_INTRO_MELODIES)}
"""

_INTRO_PROMPTS = {
    "Future Bass EDM": "Ambient future bass EDM, strong emotions, cinematic, huge crescendo buildup leading into a satisfying drop",
    "Boom Bap Hip Hop": "Boom bap hip hop. The drums in the beat are very prominent, with a hard-hitting kick and snare pattern that drives the track forward. The snare drum is particularly notable, as it has a sharp, crackling sound that gives the beat a distinctive texture. The tempo of the beat is around 84 beats per minute, which is fairly typical for a boom bap track.",
    "Upbeat EDM": "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
    "Pop Dance": "Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach",
    "Grand Orchestral": "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.",
}

_PROMPT_MESSAGE = """
Great, you've selected the melody {melody}

Next, write a prompt for your music. Try to be as detailed as possible for the best results.

Here are some pre-selected options you can use directly (click the name) or for inspiration:

{prompt_list}
"""

_GENERATING_MESSAGE = """
Generating your track from prompt: {prompt}

Reference melody: {melody_url}

{progress_indicator}
"""
_COMPLETE_MESSAGE = """
Completed! (took {seconds}s)
[Click]({output_url}) to listen to the results.

Generated from prompt: {prompt}

Reference melody: {melody_url}

You can ask me for follow-up requests to tune or change your prompt, using the "prompt: <request>" syntax (e.g. "prompt: make it more exciting").

To start again from scratch, you can clear the broom button next to the chat box to clear the context.
"""

_FOLLOW_UP_PROMPT_GENERATOR = """
You are a music generator bot that can generate a song based on a user's prompt (description of the song).

This is the user's original prompt:
{original_prompt}

After hearing the song generated from this prompt, the user had a follow-up request:
{follow_up_request}

Based on the original prompt and the follow-up request, construct the updated prompt. Detailed instructions:
* Preserve the syntax and style of the original prompt.
* Preserve the content of the original prompt that the user did not ask to change.
* The new prompt should not refer to the previous prompt or generated result. It should be a standalone prompt.

Your response will be fed programatically into the music generator, so do NOT say anything other than the new prompt.
"""


class AutoMaestro(PoeBot):
    def _get_welcome_message(self) -> ServerSentEvent:
        return self.text_event(_WELCOME_MESSAGE)

    def _get_prompt_message(self, user_input: UserInput) -> ServerSentEvent:
        return self.text_event(
            _PROMPT_MESSAGE.format(
                melody=user_input.melody_url,
                prompt_list=_gen_prompt_list(_INTRO_PROMPTS),
            )
        )

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        if len(query.query) == 1:
            yield self._get_welcome_message()
            return

        user_input = _parse_user_input(query.query[-1])

        if len(query.query) == 3:
            if user_input is None:
                user_input = UserInput(None, None)

            if not user_input.prompt:
                yield self._get_prompt_message(user_input)
                return

        if not user_input:
            yield self.text_event(_MESSAGE_FORMAT)
            return

        if user_input.melody_url and not is_url(user_input.melody_url):
            yield self.text_event(f"`{user_input.melody_url}` is not a URL.")
            return

        if user_input.melody_url is None:
            user_input.melody_url = _find_previous_melody(query.query)


        new_prompt = user_input.prompt

        # TODO: Override new_prompt with Sage-based suggestion if followup exists
        if len(query.query) >= 7:
            follow_up_request = user_input.prompt
            last_completed_generation_message = _get_last_generation_message(query.query)
            previous_prompt = _parse_previous_prompt_from_message(last_completed_generation_message)
            follow_up_prompt = await self._generate_follow_up_prompt(
                    query,
                    previous_prompt,
                    follow_up_request,
                )
            new_prompt = follow_up_prompt


        message = f"Generating your track from:\nMelody: {user_input.melody_url}\nPrompt: {new_prompt}"
        message = _GENERATING_MESSAGE.format(
            prompt=new_prompt,
            melody_url=user_input.melody_url,
            progress_indicator="Still waiting (0s elapsed...)",
        )

        yield self.text_event(message)

        generated_audio_url_task = asyncio.create_task(
            self._generate_music(user_input.melody_url, new_prompt)
        )

        i = 0
        while True:
            done, _ = await asyncio.wait(
                [generated_audio_url_task], timeout=_WAIT_TIMEOUT_S
            )
            if done:
                generated_audio_url = done.pop().result()
                break
            yield self.replace_response_event(
                _GENERATING_MESSAGE.format(
                    prompt=new_prompt,
                    melody_url=user_input.melody_url,
                    progress_indicator=f"Still waiting ({i}s elapsed...)",
                )
            )
            i += 1

        yield self.replace_response_event(_COMPLETE_MESSAGE.format(
            seconds=i,
            output_url=generated_audio_url,
            prompt=new_prompt,
            melody_url=user_input.melody_url,
        ))

    async def _generate_music(
        self,
        melody_url: Optional[str],
        prompt: Optional[str],
        override_params: Optional[dict[str, Any]] = None,
    ) -> str:

        input_params = {
            "seed": 10035627842104461000,
            "top_k": 250,
            "duration": 25,
            "temperature": 1,
            "model_version": "melody",
            "output_format": "wav",
            "continuation_end": 9,
            "continuation_start": 7,
            "continuation": False,
            "normalization_strategy": "peak",
            "classifier_free_guidance": 3,
        }
        if override_params is not None:
            input_params.update(override_params)

        if melody_url is not None:
            input_params["melody"] = melody_url

        if prompt is not None:
            input_params["prompt"] = prompt

        loop = asyncio.get_running_loop()
        output_url = await loop.run_in_executor(
            None,
            lambda: replicate.run(
                "joehoover/musicgen:ba9bdc5a86f60525ba23590a03ae1e407b9a40f4a318a85af85748d641e6659f",
                input=input_params,
            ),
        )
        return output_url


    async def _generate_follow_up_prompt(
        self,
        query: QueryRequest,
        original_prompt: str,
        follow_up_request: str,
    ) -> str:
        llm_input_prompt = _FOLLOW_UP_PROMPT_GENERATOR.format(
            original_prompt=original_prompt,
            follow_up_request=follow_up_request,
        )

        protocol_messages = [
            ProtocolMessage(role="user", content=llm_input_prompt)
        ]
        chatgpt_query = QueryRequest(
            version=query.version,
            type=query.type,
            query=protocol_messages,
            user_id=query.user_id,
            conversation_id=query.conversation_id,
            message_id=query.message_id,
            api_key=query.api_key,
        )

        msg = await get_final_response(chatgpt_query, "chatGPT", chatgpt_query.api_key)
        return msg



if __name__ == "__main__":
    run(AutoMaestro())
