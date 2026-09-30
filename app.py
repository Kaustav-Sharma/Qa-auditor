import io
import re
import streamlit as st
import extra_streamlit_components as stx
from pydub import AudioSegment
from groq import Groq

st.set_page_config(page_title="Call QA Auditor", layout="wide")

# Initialize Cookie Manager for Scenario B (Browser Cookies)
cookie_manager = stx.CookieManager()

st.title("🎙️ AI QA Auditor")
st.caption("Upload stereo audio to generate a timestamped transcript, emotion tags, and an automated QA Scorecard.")

with st.sidebar:
    st.header("Settings")
    
    # Retrieve saved key from browser cookies
    saved_key = cookie_manager.get(cookie="groq_api_key")
    current_val = saved_key if saved_key is not None else ""
    
    groq_api_key = st.text_input("Enter Groq API Key:", value=current_val, type="password")
    
    # Save the key to the browser if a new one is typed
    if groq_api_key and groq_api_key != saved_key:
        cookie_manager.set("groq_api_key", groq_api_key)
        
    st.markdown("[Get a free Groq API key here](https://console.groq.com/keys)")

uploaded_files = st.file_uploader(
    "Upload audio files (.wav, .mp3, .mpeg)", 
    type=["wav", "mp3", "mpeg"], 
    accept_multiple_files=True
)

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def process_channel(audio_segment, client, channel_name):
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3", bitrate="16k") 
    buffer.name = f"{channel_name}.mp3"
    buffer.seek(0)
    
    response = client.audio.transcriptions.create(
        file=(buffer.name, buffer.read()),
        model="whisper-large-v3",
        prompt="you are an expert call transcriber, transcribe calls perfectly",
        language="hi",
        temperature=0.1,
        response_format="verbose_json"
    )
    
    segments = []
    if hasattr(response, 'segments') and response.segments:
        for seg in response.segments:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": channel_name,
                "text": seg["text"].strip()
            })
    return segments

def call_groq_text_model(prompt, client, required_char, temperature=0.0, presence_penalty=0.0):
    try:
        models_response = client.models.list()
        chat_models = [
            m.id for m in models_response.data 
            if "whisper" not in m.id.lower() and "base" not in m.id.lower()
        ]
        
        def score_model(m_id):
            score = 0
            m_id = m_id.lower()
            if "versatile" in m_id or "instant" in m_id or "-it" in m_id: score += 10
            if "70b" in m_id or "large" in m_id: score += 5
            return score
            
        chat_models.sort(key=score_model, reverse=True)
    except Exception as e:
        raise Exception(f"Failed to fetch model list from Groq API: {str(e)}")

    if not chat_models:
        raise Exception("No valid text models found for your Groq API key.")

    last_error = ""
    for model_name in chat_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                presence_penalty=presence_penalty,
                max_tokens=4000
            )
            output = response.choices[0].message.content.strip()
            
            if output.count("[Neutral]") > 15 and len(output) < 300:
                last_error = "Repetition loop detected."
                continue

            if required_char in output and len(output) > 10:
                return output
                
        except Exception as e:
            last_error = str(e)
            continue
            
    raise Exception(f"All dynamic text models failed. Last error received: {last_error}")

def tag_emotions(transcript_text, client):
    prompt = f"""
    You are an expert conversational analyst. Rewrite the exact transcript below, inserting an emotion tag (e.g., [Neutral], [Frustrated], [Angry], [Relieved]) directly after the speaker's name.
    
    CRITICAL INSTRUCTION: You MUST output the full text of the conversation. DO NOT just output a list of tags. 
    Do not add any introductions or summaries. Just output the tagged transcript.

    TRANSCRIPT TO PROCESS:
    {transcript_text}
    """
    return call_groq_text_model(prompt, client, required_char="[", temperature=0.2, presence_penalty=0.5)

def generate_scorecard(tagged_transcript, client):
    prompt = f"""
    You are a Quality Assurance Auditor. Evaluate the following emotion-tagged transcript based on the exact rubric below.
    
    STRICT SCORING RULES - READ CAREFULLY:
    1. DEFAULT TO MAX SCORE: You MUST award the MAXIMUM possible score for every single parameter by default. 
    2. BURDEN OF PROOF & TRANSCRIPT ERRORS: AI transcripts often cut off mid-sentence or miss words. You are strictly FORBIDDEN from deducting marks for incomplete sentences or assuming the agent failed just because the transcript is fragmented. If you deduct marks, your justification MUST contain a direct quote proving a clear agent failure. If no absolute negative proof exists, you MUST award the max score.
    3. EXACT MARKS ONLY: You are ONLY permitted to use the exact numeric values listed below.
    4. GRAMMAR & PUNCTUATION (Param 7): The transcription AI naturally makes punctuation and grammar mistakes. Do NOT deduct marks for missing punctuation or minor grammatical errors. Only deduct if there is undeniable proof of a massive, unprofessional linguistic failure by the agent.

    PARAMETER SPECIFIC DEFINITIONS:
    - Opening Protocol (Param 1): The agent must completely finish their opening greeting within the first 10 seconds of the call (by 00:10). If the opening spills past 10 seconds, deduct marks.
    - Took Ownership (Param 3): The agent proactively checks information without forcing the customer to repeat details they already provided. Note: Asking a question to *confirm* details is good practice and MUST NOT be penalized.
    - Clarity / Fumbling (Param 6): Fumbling is strictly defined as the agent using repeated filler words. Only deduct marks if there is clear evidence of the agent relying on repeated filler words.
    - Empathy and Sympathy (Param 8): Do not deduct marks just because the agent failed to use specific empathetic phrases. Only deduct marks if the agent actively misbehaves, is dismissive, or is rude to the customer.
    - Probing (Param 10): Asking specific, necessary questions. Only deduct marks if probing was clearly required to resolve the issue and the agent completely failed to ask.
    - Interruption & Assurance (Param 12): Assurance means giving the customer a timeline or guarantee of when something will happen. Deduct marks if the situation called for a timeline or guarantee and the agent failed to provide one.
    - Closing Protocol (Param 15): The agent MUST ask if there is anything else they can assist the customer with, AND they MUST say "thank you" (or a direct equivalent) at the very end of the call. Deduct marks if either of these two elements is missing.

    RUBRIC & ALLOWED SCORES:
    1. Opening Protocol followed (Allowed Marks: 0 or 3)
    2. Followed Customer's Language (Allowed Marks: 0 or 2)
    3. Took Ownership/ Went extra Miles (Allowed Marks: 0, 5, or 10)
    4. Was the responses aligned with cx concern? (Allowed Marks: 0, 3, or 5)
    5. Was polite and Courteous throughout the Call (Allowed Marks: 0, 5, or 10)
    6. Had CLARITY of voice/fumbling/rate of speech (Allowed Marks: 0, 3, or 6)
    7. Grammatical/Pronunciation Error (Allowed Marks: 0, 3, or 5)
    8. Empathy and Sympathy (Allowed Marks: 0, 4, or 8)
    9. Avoided Dead Air (Allowed Marks: 0 or 3)
    10. Was probing done on the call/Proactivness (Allowed Marks: 0, 4, or 8)
    11. Followed hold & mute protocol (Allowed Marks: 0, 5, or 10)
    12. Interruption & Assurance (Allowed Marks: 0, 5, or 10)
    13. Effective listening (Allowed Marks: 0, 3, or 5)
    14. Process knowledge/Tat (Allowed Marks: 0, 5, or 10)
    15. Followed closing protocol (Allowed Marks: 0, 3, or 5)

    Output a strict Markdown table with columns: | Parameter | Score Awarded | Max Score | Justification (Must include exact quote if marks are deducted) |
    At the very bottom, bold the Final Total Score out of 100. Do not include any other text.

    TRANSCRIPT:
    {tagged_transcript}
    """
    return call_groq_text_model(prompt, client, required_char="|")

def extract_final_score(scorecard_text):
    matches = re.findall(r'(?i)total.*?score.*?(\d{1,3})', scorecard_text)
    if matches:
        return int(matches[-1])
    
    matches = re.findall(r'\*\*(\d{1,3})(?:/100)?\*\*', scorecard_text)
    if matches:
        return int(matches[-1])
        
    return None

if st.button("Audit Calls", type="primary"):
    if not groq_api_key:
        st.error("Please enter a valid Groq API key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one audio file.")
    else:
        client = Groq(api_key=groq_api_key)
        
        for file in uploaded_files:
            st.divider()
            st.subheader(f"📁 File: {file.name}")
            
            attempt = 1
            max_attempts = 5
            
            emotion_transcript = ""
            scorecard = ""
            
            while attempt <= max_attempts:
                if attempt > 1:
                    st.info(f"🔄 Rerunning process for {file.name} (Attempt {attempt}/{max_attempts})")
                
                raw_transcript = ""
                
                with st.spinner(f"Attempt {attempt}: 1/3 Transcribing via Groq..."):
                    try:
                        file.seek(0)
                        
                        audio = AudioSegment.from_file(file)
                        if audio.channels < 2:
                            st.error(f"'{file.name}' is mono. Stereo audio is required.")
                            break
                        
                        channels = audio.split_to_mono()
                        agent_segs = process_channel(channels[0], client, "Agent")
                        customer_segs = process_channel(channels[1], client, "Customer")
                        
                        full_timeline = sorted(agent_segs + customer_segs, key=lambda x: x["start"])
                        
                        transcript_lines = []
                        for item in full_timeline:
                            t_start = format_timestamp(item['start'])
                            t_end = format_timestamp(item['end'])
                            line = f"[{t_start} - {t_end}] {item['speaker']}: {item['text']}"
                            transcript_lines.append(line)
                        
                        raw_transcript = "\n".join(transcript_lines)
                    except Exception as e:
                        st.error(f"Error during audio processing: {str(e)}")
                        break
                
                if raw_transcript:
                    with st.spinner(f"Attempt {attempt}: 2/3 Tagging emotions..."):
                        try:
                            emotion_transcript = tag_emotions(raw_transcript, client)
                        except Exception as e:
                            st.error(f"Error during emotion tagging: {str(e)}")
                            emotion_transcript = raw_transcript
                    
                    with st.spinner(f"Attempt {attempt}: 3/3 Auditing scorecard..."):
                        try:
                            scorecard = generate_scorecard(emotion_transcript, client)
                        except Exception as e:
                            st.error(f"Error generating scorecard: {str(e)}")
                            scorecard = "Scorecard generation failed."

                    final_score = extract_final_score(scorecard)
                    
                    if final_score is not None:
                        if final_score == 100 or final_score < 90:
                            if attempt < max_attempts:
                                st.warning(f"Score was {final_score}. Rerunning to ensure accuracy...")
                            else:
                                st.warning(f"Max attempts reached. Final score settled at {final_score}.")
                            attempt += 1
                        else:
                            st.success(f"Final Score accepted: {final_score}")
                            break 
                    else:
                        st.warning("Could not automatically parse the score number. Stopping retries.")
                        break
                else:
                    break 

            if emotion_transcript and scorecard:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Emotion-Tagged Transcript")
                    st.text_area("Transcript", emotion_transcript, height=500, label_visibility="collapsed")
                    st.download_button(
                        label="⬇️ Download Transcript",
                        data=emotion_transcript,
                        file_name=f"{file.name}_transcript.txt",
                        mime="text/plain",
                        key=f"dl_tx_{file.name}"
                    )
                
                with col2:
                    st.markdown("### QA Evaluation Scorecard")
                    
                    ui_display_scorecard = []
                    for line in scorecard.split('\n'):
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 5:
                                clean_line = f"| {parts[1].strip()} | {parts[2].strip()} | {parts[3].strip()} |"
                                ui_display_scorecard.append(clean_line)
                            else:
                                ui_display_scorecard.append(line)
                        else:
                            ui_display_scorecard.append(line)
                    
                    st.markdown("\n".join(ui_display_scorecard))
