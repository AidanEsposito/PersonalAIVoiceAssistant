import whisper
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
import os



load_dotenv() # Load OPENAI Key from .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def recognize_speech_from_microphone(filename="input.wav"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak something...")
        audio = r.listen(source)
        try:
            print("Recognizing...")
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            return filename
             
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None




def transcribe(filename):
    #Load the Whisper model
    model = whisper.load_model("base") 

    # Transcribe the audio file
    print("Transcribing audio file...")
    audio_result = model.transcribe(filename, language="en", verbose=True)

    #Print the audio results
    print("Transcription Result:")
    print(audio_result["text"])

def chat_with_gpt(audio_result):
    #Send to ChatGPT and get a response
    print("Sending audio to ChatGPT...")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": audio_result}
        ]
    )
    reply = response.choices[0].message.content
    print("ChatGPT:", reply)
    return reply


# Speech to text and chat with GPT
if __name__ == "__main__":
    audio_file = recognize_speech_from_microphone()
    if audio_file:
        text = transcribe(audio_file)
        if text:
            chat_with_gpt(text)
