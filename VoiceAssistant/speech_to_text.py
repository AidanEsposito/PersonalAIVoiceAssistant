import whisper
import speech_recognition as sr

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