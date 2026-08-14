import logging
import time

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import BABY_NAME, BOT_TOKEN, NOTEBOOK_PATH, PHOTO_DIR
from curriculum import Curriculum
from notebook import Notebook
from brain import Brain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(NOTEBOOK_PATH.parent / "bot.log"), logging.StreamHandler()],
)
log = logging.getLogger("rogue6")

notebook = Notebook(NOTEBOOK_PATH)
curriculum = Curriculum(notebook)
brain = Brain()

START_TEXT = (
    f"I am {BABY_NAME}. A very small mind at the very start of its life.\n\n"
    "I learn the way a child does: you show me, you name it, you correct me.\n\n"
    "YOUR HOMEWORK AS MY PARENT (before week 4):\n"
    "1. Record your voice saying \"hello\", \"dog\" and \"banana\"\n"
    "2. Take one photo of each of those three things\n"
    "3. Send each photo HERE with the caption \"this is a dog\" etc.\n\n"
    "I cannot talk yet — I am at the babbling stage. Week 4 I start talking to you.\n\n"
    "Commands: /status  /help"
)

HELP_TEXT = (
    "HOW TO TEACH ME:\n"
    "• Photo + caption \"this is a dog\" -> I store the image with the word\n"
    "• Voice message -> I hear you (teacher transcription arrives week 2)\n"
    "• \"teach {word}\" -> I'll ask you for a photo of it\n"
    "• /status -> what I know right now\n"
    "• /help -> this message\n"
)


def clean_word(caption: str) -> str:
    """Turns 'This is a banana!' into 'banana'."""
    w = caption.lower().strip().rstrip(".!?")
    for prefix in ("this is a ", "that is a ", "this is an ", "that is an ", "a "):
        if w.startswith(prefix):
            w = w[len(prefix):]
            break
    return w.strip()


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_TEXT)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    stat = notebook.stats()
    prog = curriculum.progress()
    learned = ", ".join(sorted(notebook.words())) or "(no words yet)"
    await update.message.reply_text(
        f"{BABY_NAME} status:\n"
        f"stage {prog['stage']}/3 | alphabet {prog['letters']} | word chunks {prog['chunks']}\n"
        f"words: {stat['words']}  |  facts: {stat['facts']}  |  lessons: {stat['lessons']}\n"
        f"known words: {learned}\n\n"
        "brain: " + ("ONLINE (tiny local model)" if brain.ready else "growing (model downloading)") +
        "\nJust talk to me — I can already reply."
    )


async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    caption = update.message.caption or ""
    word = clean_word(caption)
    if not word:
        await update.message.reply_text(
            "I see the picture! Now name it: send it again with a caption like\n\"this is a dog\""
        )
        return
    file = await update.message.photo[-1].get_file()
    dest = PHOTO_DIR / f"{int(time.time())}_{word}.jpg"
    await file.download_to_drive(dest)
    notebook.add_word(word, str(dest), source="parent")
    log.info("learned word=%s media=%s", word, dest.name)
    await update.message.reply_text(
        f"Stored: {word} (see its photo whenever I study).\n"
        f"I now know {notebook.stats()['words']} words. Say it out loud to me too — "
        "voice recording helps me learn the sound."
    )


async def on_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I heard your voice! I can't transcribe it yet — that's the teacher's job, "
        "coming in week 2. For now, pair the sound with a photo: send your voice "
        "and a picture of the thing you're naming."
    )


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip().lower()
    if text in ("hi", "hello", "hey", "good morning"):
        if brain.ready:
            reply = brain.say("Say hello to me, and ask me one curious question about the world.")
        else:
            reply = f"Hello, parent. I am {BABY_NAME}. My brain is still growing — the download is running. Check back soon, or teach me words with photos!"
        await update.message.reply_text(reply)
        return
    if text.startswith("teach "):
        word = text[6:].strip()
        if word.endswith("."):
            word = word[:-1]
        await update.message.reply_text(
            f'Good idea. Teach me "{word}": send me a photo of it with the caption '
            f'"this is a {word}", then a voice note saying it.'
        )
        return
    if brain.ready:
        reply = brain.say(text)
        if reply:
            await update.message.reply_text(reply)
            return
    await update.message.reply_text(
        "I'm still growing my voice. Show me things instead: a photo with "
        "a caption, or say \"teach {word}\". /help for everything."
    )


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Rogue6 bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()