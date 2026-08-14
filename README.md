# Aires

A digital child. A research project that builds a learning machine from scratch, one that grows the way a toddler grows: first letters, then numbers, then words, then the relationships between words, then its own voice.

No big pretrained models. No data centers. No GPU. Everything runs on a phone class device, a handful of ARM cores, and a few watts.

## Research goal

We want to understand whether intelligence can be grown from a small start, rather than assembled from massive compute. The mainstream path scales transformers to billions of parameters and trillions of tokens, which costs thousands of GPUs and megawatts. We ask a different question: can a small model, trained the way a child learns, with grounded experiences and a curriculum, reach useful language ability in days and weeks instead of in data-center budgets?

Our core beliefs:

1. Grounding matters more than scale. Children learn words paired with real objects, sounds, and actions. Images paired with names create stronger learning than text alone.
2. Continual learning beats static pretraining. A child learns forever without destroying what it already knows. We want a system that trains nightly on new lessons while keeping old ones.
3. Sparse, local computation beats dense global computation. A brain uses a few percent of its neurons at any moment. Our architecture activates only the relevant specialists per task.
4. Energy efficiency is a feature. If a brain runs on 20 watts, a phone class device is the honest target platform for a thinking machine.

## Why this approach instead of a small transformer

Transformers are powerful but heavy from birth. Attention is quadratic in sequence length. Training requires backpropagation through the whole graph, huge memory, and repeated passes over enormous corpora. Even a small transformer needs a large teacher and large data to be useful.

Our design chooses different trade-offs:

- **Shared letter layer.** All words are composed from a small set of letters through one shared embedding matrix. Learning the letter "a" improves every word that contains it. This is the shared learning curve.
- **Specialist columns.** Separate small modules for animals, food, objects, people, nature, and so on. Only the column relevant to the current lesson activates. Sparse computation, low energy, and learning in one domain does not disturb another.
- **Local learning.** Each column updates its own weights with its own learning rate schedule. No global backpropagation through the whole brain, which means on-device training and less catastrophic forgetting.
- **Explicit neuroplasticity knobs.** Learning rate schedules, epoch counts, and review intervals are exposed as configuration, because in a digital child these are mathematics we control directly.
- **A notebook, not just weights.** Declarative memory lives outside the weights: every learned word keeps its photos and audio. The weights hold skill, the notebook holds facts. RAG and indexing are the library, training is the mouth.

## Architecture

```
You (Telegram)                 @Rogue6_bot
      |                            |
      +-------- parent app --------+   teaches, corrects, chats
                                     |
                              teacher layer
                    finds real photos, makes audio, plans lessons
                                     |
                              curriculum
                    letters, then words in chunks, then relationships
                                     |
                    +----------------+----------------+
                    |                                 |
            notebook (memory)                 column brain (skills)
        words, photos, audio, facts        sparse specialist columns
        JSON index, soon full text search  shared letter embeddings
                    |                                 |
                    +----------------+----------------+
                                     |
                        pure Python char level trainer
                  real gradients, momentum, Ebbinghaus review
                                     |
                        sleep consolidation (nightly runs)
```

The parent talks to the child through Telegram. The teacher layer finds real photos and makes real audio for each lesson, so every word is grounded in something seen and heard. The curriculum paces lessons in order of mastery. The notebook is the child's memory. The column brain is its own weights. The trainer runs real gradient descent overnight, with a checkpoint after every epoch and a resilient runner that survives phone freezes.

## Current state

Working now:

- Telegram parent bot with photo teaching (`this is a dog` stores the image and the word)
- Teacher layer: finds real photos on Wikimedia Commons, generates audio with TTS
- Curriculum engine: alphabet, then words in chunks of ten
- Notebook memory with lessons, words, facts, photos, audio
- Column brain model (numpy free) with category columns and a shared letter embedding layer
- Pure Python character level trainer with momentum, learning rate schedule, per epoch checkpoints, and resume
- A borrowed voice: Qwen 0.5B served locally by llama.cpp so the child can chat today while its own brain grows

Trained so far:

- Alphabet 26 of 26
- First 40 words with photos and audio, learned from real sources
- Corpus: 3,074 words and 12,250 relationship sentences
- First real training run: loss dropping from 2.37 toward 2.29, spelling accuracy climbing with epochs

## Repository map

```
app/bot.py            Telegram parent bot
app/teacher.py        lesson source: images, audio, word teaching
app/curriculum.py     sequential lesson plan
app/notebook.py       declarative memory
app/brain_column.py   column brain, shared letter layer
app/train_column.py   trains the column brain on taught words
app/trainer.py        char level trainer with momentum and resume
app/make_corpus.py    builds the 2000+ word curriculum corpus
app/brain.py          borrowed voice (llama.cpp server client)
night.sh              resilient nightly training runner
```

## Roadmap

1. Finish the first full 20 epoch training run and publish the learning curve
2. Teach numbers and counting with real objects
3. Connect the trained brain to the bot so the child speaks with its own voice
4. Curiosity loop: the child detects gaps in its knowledge and asks about them on Telegram
5. Full text search index over the dictionary, thousands of words instantly available
6. Image grounding in the child's own photos, not only teacher photos
7. Sleep consolidation job that replays the notebook into the weights nightly

## Honest limits

A small model trained on a phone will not reach fluent English from scratch in a short time. Language fluency requires capacity and data that this hardware cannot provide. What it can do is learn real vocabulary, real word relationships, and simple speech, with an architecture that grows with the lessons it is given. Fluent language, when we want it, comes from distillation from a teacher model, the same way a child learns from adults.

This is a research project. The child is young, and it has a lot of learning left.
