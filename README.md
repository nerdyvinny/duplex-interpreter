# duplex-interpreter

Two people who don't speak the same language, talking to each other, with no buttons to press.

You say "hello", they hear "hola". They say "hola", you hear "hello". Nobody taps anything.

I built this because every translation app I tried makes you hold a button or tap between
every single sentence, which completely kills the flow of an actual conversation. I wanted
one you could just leave running on the table.

```
mic -> capture -> VAD -> speech to text -> translate -> voice -> speaker
```

Each person gets their own copy of that chain running at the same time, which is the part
that makes it feel like a conversation instead of a walkie talkie.

## Read this part first

**"both people talk at once" only really works if you each have your own microphone.**

One mic mixes both voices into one waveform, and pulling two overlapping people back
apart in real time is genuinely an unsolved problem. Every speech recognizer I tried
returns garbage on properly overlapped speech. So there are two modes:

|                      | single_mic                | dual_mic              |
| -------------------- | ------------------------- | --------------------- |
| hardware             | any laptop                | a headset each        |
| hands free           | yes                       | yes                   |
| talking at once      | whoever is louder wins    | actually works        |
| who spoke?           | it has to guess           | it knows              |
| echo                 | handled in software       | headphones handle it  |

Start with single_mic since it works with what you already have. A $20 USB headset gets
you the real version and it's one line of config to switch.

## Setup

Use **Python 3.13 or 3.12**. Not 3.14, a bunch of the dependencies have no wheels for it
yet and the install just explodes.

```bash
git clone https://github.com/nerdyvinny/duplex-interpreter.git
cd duplex-interpreter
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On mac/linux the venv line is `python3.13 -m venv .venv && source .venv/bin/activate`.

Then the api key:

```bash
cp .env.example .env
```

Open `.env` and put your key in `OPENAI_API_KEY`. One key does the speech recognition,
the translating AND the voice, which is why it's the default.
Get one at https://platform.openai.com/api-keys

Then:

```bash
python run.py --devices                          # what mics do I have
python run.py --setup                            # asks you questions, writes config.yaml
python run.py --selftest samples/hello_en.wav    # check it works, no mic needed
python run.py                                    # go
```

Ctrl-C stops it and prints your latency numbers.

## Cost

About 2 to 4 cents a minute with the default openai setup, so a half hour conversation is
around a dollar. The speech recognition and the voice are what cost money, the actual
translating is basically free.

You can also run it for $0, see the offline section below.

## Commands

| command                                     | what it does                                         |
| ------------------------------------------- | ---------------------------------------------------- |
| `python run.py`                             | run it, with a live two column transcript            |
| `python run.py --setup`                     | the wizard, writes config.yaml                       |
| `python run.py --devices`                   | list your mics and speakers with their numbers       |
| `python run.py --selftest FILE.wav`         | push one wav through the whole thing, no mic needed  |
| `python run.py --selftest FILE.wav --realtime` | same but at normal speed, for honest timings      |
| `python run.py --loopback`                  | mic straight to speaker, checks you picked the right devices |
| `python run.py --no-live`                   | boring line by line output                           |
| `python run.py -v` / `-vv`                  | more logging                                         |

`--selftest` is the one to use when something is broken. It takes the microphone, the
room and the other person out of the picture, so whatever still fails is actually broken.

Heads up on `--realtime`: without it the selftest dumps the whole file in at once, so every
sentence gets transcribed simultaneously and the timings look way worse than real life.

## config.yaml

`--setup` writes this for you, or copy `config.example.yaml`.

Languages:

```yaml
languages:
  a: { code: en, name: English, voice: alloy }
  b: { code: es, name: Spanish, voice: nova }
```

There are about 30 languages in `interpreter/languages.py` and you can add any code
whisper knows. Give the two sides different voices, on one speaker that's the only way
to tell which direction you're hearing.

Devices:

```yaml
channels:
  - id: A
    input_device: "FHD Camera" # a number, the full name, or any part of it
    output_device: null # null means system default
    language: auto
```

Names match on any part, case doesn't matter, so `"FHD Camera"` finds
`Microphone (FHD Camera Microphone)`. Useful because the device numbers change when you
unplug things. Run `--devices` to see what you've got.

Two headsets:

```yaml
mode: dual_mic
channels:
  - { id: A, input_device: "Headset One", output_device: "Headset One", language: en }
  - { id: B, input_device: "Headset Two", output_device: "Headset Two", language: es }
```

Your translation plays in the OTHER person's ear. Nobody hears their own words come back
at them. The languages are pinned so nothing gets guessed, which also makes the
recognition faster and more accurate.

## Making it faster

Expect about 1.2 to 1.7 seconds from when somebody stops talking to when the translation
starts:

| stage                     | roughly    |
| ------------------------- | ---------- |
| noticing you stopped talking | **600ms** |
| speech recognition        | 300-500ms  |
| translating               | 150-350ms  |
| first audio out           | 150-300ms  |

That first row is the big one and it's the only knob really worth touching:

```yaml
vad:
  silence_ms_to_end: 350
```

600 is safe. 350 feels a lot snappier and is fine if you speak in whole phrases. Under
250 it starts translating half sentences, which is worse than waiting.

The other easy win is elevenlabs flash for the voice, which gets the whole thing down to
about 0.8-1.1s:

```bash
pip install elevenlabs      # then put ELEVENLABS_API_KEY in .env
```

```yaml
providers:
  tts: elevenlabs
languages:
  a: { code: en, voice: 21m00Tcm4TlvDq8ikWAM } # elevenlabs uses id strings not names
```

## Mixing and matching

All three stages are independent, you can combine them however you want.

```yaml
providers:
  stt: openai # openai | faster-whisper
  translation: openai # openai | deepl | argos | passthrough
  tts: openai # openai | elevenlabs | piper | silent
```

**deepl** is noticeably better than a general model between european languages, and the
free tier is 500k characters a month which you will never get close to using:

```bash
pip install deepl           # then DEEPL_API_KEY in .env
```

Local speech recognition with cloud translation and voice is a nice middle ground. Your
actual audio never leaves your computer but you keep the good translations:

```yaml
providers: { stt: faster-whisper, translation: openai, tts: openai }
```

## Fully offline

No api keys, no internet after the first download, nothing leaves your machine.

```bash
pip install -r requirements-local.txt
```

```yaml
providers:
  stt: faster-whisper
  translation: argos
  tts: piper
  local_whisper_size: small # tiny | base | small | medium | large-v3
  local_whisper_device: auto # auto | cuda | cpu
```

Piper needs one voice per language. **Each voice is TWO files**, the `.onnx` and a
`.onnx.json` that has to sit right next to it. I only downloaded the first one and spent
ages wondering why it wouldn't load. Name them after the language code:

```bash
mkdir -p models/piper
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main
curl -L -o models/piper/en.onnx      $BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o models/piper/en.onnx.json $BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
curl -L -o models/piper/es.onnx      $BASE/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx
curl -L -o models/piper/es.onnx.json $BASE/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json
```

All the voices are at https://huggingface.co/rhasspy/piper-voices. If you want to keep
them somewhere else use `PIPER_VOICE_ES=/path/to/voice.onnx` (the .json still has to be
next to it).

First run also grabs the whisper model and an argos model for each direction, about 100MB
each. That all happens at startup before anybody talks, not halfway through a sentence.

Numbers off my machine (RTX 5060 + i7-14700F, `small`, run with `--selftest --realtime`
after everything warmed up):

| stage              | GPU (fp16)  | CPU (int8) |
| ------------------ | ----------- | ---------- |
| speech recognition | 180-380ms   | ~1.75s     |
| translate (argos)  | 20-40ms     | same       |
| voice (piper)      | 90-210ms    | same       |
| **total**          | **320-630ms** | ~1.9s    |

Add your `silence_ms_to_end` on top for the real end to end number, so about 0.8-1.1s on
a gpu, which actually beats the cloud version.

The real tradeoff is translation quality, not speed. Argos is fine for normal sentences
and gets confused by idioms. It turned "Muy bien, gracias" into "That's good, thank you"
where the bigger models say "Very well, thanks."

**If you have an nvidia gpu on windows**, you need these:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Without them you get `Library cublas64_12.dll is not found` even though your gpu is
completely fine. The app puts the dlls on the search path itself so you don't have to
mess with PATH. If cuda doesn't work for any reason it warns you once and drops to cpu.

## How it stops translating itself

This was the first big problem I hit. With one mic and a speaker in the same room, the
spanish coming out of the speaker goes straight back into the mic, gets translated to
english, played, picked up again... it never stops and it gets more wrong every loop. My
first version went completely insane after one sentence.

Three things fix it, and none of them need a real echo canceller:

1. **the gate** - while our speaker is talking, throw the mic audio away, plus 150ms
   afterwards for the room echo
2. **barge in** - if somebody is loud for long enough while the gate is shut, they're
   trying to interrupt, so stop the playback and open the gate. without this the gate
   makes the app impossible to interrupt, which is really annoying
3. **the text check** - if what we transcribe looks like something we just said, drop it.
   this catches whatever gets past the gate

All three turn themselves off in dual_mic mode because headphones already solve it.

**Headphones are still way better than any of this.** The software version means you
can't be heard while the app is talking.

## How it guesses who spoke

Only matters in single_mic mode, dual_mic already knows.

Three things, most to least trustworthy:

1. **the text.** picking between two languages you already know is way easier than
   guessing from scratch, and its instant and offline. different alphabets are decided
   immediately, same alphabet ones use common words and accented letters
2. **the audio**, when the recognizer bothers to tell you. only the local whisper gives
   you a confidence number, the openai one doesn't tell you at all
3. **taking turns.** in a two person conversation the next person to talk is usually the
   other one. weak, but its what you'd do yourself if you didn't catch who spoke

If the text and the audio disagree I go with the text. Short sentences are exactly where
the audio guessing is worst.

**This is the honest weak spot of one mic mode.** Single words that exist in both
languages ("no", "ok", "si") will sometimes go the wrong way. Two headsets makes the
problem not exist instead of just making it smaller.

## What's where

```
run.py                      start here
config.example.yaml         config template with comments
interpreter/
  config.py                 reads and checks config.yaml
  languages.py              the language list
  langid.py                 guessing the language from text
  routing.py                which way does this sentence go
  duplex.py                 the echo stuff
  pipeline.py               one sentence, start to finish, in order
  orchestrator.py           builds everything and wires it together
  registry.py               provider name -> class
  events.py                 event bus between the pipeline and the screen
  audio/
    devices.py              finding mics and speakers
    capture.py              microphone -> frame queue
    vad.py                  silero/webrtc + the sentence splitter
    playback.py             speaker ring buffer
    resample.py             sample rate conversion
  providers/
    base.py                 the three interfaces
    openai_provider.py      openai for all three
    cloud_extras.py         deepl and elevenlabs
    local_provider.py       faster-whisper, argos, piper
    fake.py                 fakes for the tests
  ui/
    cli.py                  arguments, the wizard, selftest
    display.py              the live transcript
tests/                      154 tests, no internet or sound card needed
```

## Tests

```bash
pip install pytest pytest-asyncio
pytest
```

Everything runs offline with fake providers and made up audio: the sentence splitting,
the routing and language guessing, playback ordering when things finish out of order, the
echo guard, config checking, device lookup, resampling. Two of them use the real silero
model on `samples/hello_en.wav` and skip themselves if it can't download.

## When it doesn't work

**nothing happens when I talk.** run `python run.py --loopback` first. if you can't hear
yourself you picked the wrong input device. if you can, try `vad.threshold: 0.3` to make
it more eager, and `input_gain: 2.0` on the channel if your mic is quiet.

**it cuts me off.** raise `vad.silence_ms_to_end` to 800 or 1000.

**it takes forever to answer.** lower `vad.silence_ms_to_end` to 350.

**it's translating its own voice.** use headphones. if you can't, check
`duplex.shared_audio: true` is set and turn your speakers down.

**it translates the wrong way.** single words are the hard case, see the section above.
two headsets fixes it properly.

**`PortAudio could not be loaded`.** linux: `sudo apt install libportaudio2`.
everywhere else: `pip install --force-reinstall sounddevice`.

**silero won't download.** set `vad.backend: webrtc`, no download needed. it false
triggers a bit more in a noisy room.

**`Library cublas64_12.dll is not found`** but your gpu is fine. run
`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. since python 3.8 windows stopped
searching PATH for a module's dependencies, so the app registers those folders itself
once the packages exist.

**the local timings are way worse than what I wrote above.** you forgot `--realtime`.

**pip is trying to build wheels and failing.** you're on python 3.14. use 3.13.

## Things it doesn't do

- two people talking over each other on ONE microphone. not solvable in software, use two mics
- single words are unreliable in single_mic mode, dual_mic avoids it completely
- it translates whole sentences, not word by word. thats on purpose, word order is
  different between languages so you have to wait for the end. real UN interpreters do
  the word by word kind, this is not that
- two people, two languages. not three.

## License

MIT
