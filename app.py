from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from typing import Any, Dict, List, Optional, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult
import os
import sys
import logging
import json
import openai
from langchain.chat_models import ChatOpenAI
import azure.cognitiveservices.speech as speechsdk
import asyncio
from flask import Flask, render_template, request, send_file, abort, send_from_directory,request, Response
from flask_socketio import SocketIO
import base64
from flask_cors import CORS
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, audio
from datetime import datetime
import uuid
import os
from threading import Timer
import logging
from queue import Queue

logging.basicConfig(level=logging.INFO)  # INFOレベル以上のログを出力

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


openai.api_key = os.environ['OPENAI_API_KEY']
azure_speech_key = os.environ['SPEECH_KEY']
azure_service_region = os.environ['SERVICE_REGION_KEY']

listeners = []  # クライアントのリスナーを追跡

def send_audio_url_to_client(url):
    for listener in listeners:
        listener.send(url)

@app.route('/events')
def events():
    def stream():
        message_queue = Queue()
        listeners.append(message_queue)
        try:
            while True:
                message = message_queue.get()
                yield f"data: {json.dumps({'url': message})}\n\n"
        except GeneratorExit:  # クライアント接続が閉じた場合
            listeners.remove(message_queue)

    return Response(stream(), content_type='text/event-stream')



def delete_file(path):
    """指定されたファイルを削除する関数、ファイルが存在する場合のみ"""
    if os.path.exists(path):  # ファイルが存在するかチェック
        try:
            os.remove(path)
            logger.info(f"Deleted file: {path}")
        except Exception as e:
            logger.error(f"Error deleting file: {path}, {e}")
    else:
        logger.info(f"File does not exist, no need to delete: {path}")

def text_to_speech(text):
    output_filename = f"text2output_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}_{uuid.uuid4().hex}.wav"
    output_dir = 'audio'  # 音声ファイルを保存するディレクトリ
    output_path = os.path.join(output_dir, output_filename)  # 音声ファイルの保存パス

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # SSMLを使用してテキストを音声に変換
    ssml_string = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="ja-JP">
                    <voice name="ja-JP-DaichiNeural">
                        <mstts:express-as style="customerservice" styledegree="3">
                            {text}
                        </mstts:express-as>
                    </voice>
                  </speak>"""
    result = synthesizer.speak_ssml_async(ssml_string).get()

    # ファイル削除のためのタイマーを設定（2分後に削除）
    Timer(120, delete_file, args=[output_path]).start()

    return result, output_filename

buffered_text = ""

@app.route('/audio/<filename>')
def audio_file(filename):
    """音声ファイルを提供するルート"""
    return send_from_directory('audio', filename)

def dummy_callback(token):
    global buffered_text
    buffered_text += token
    print(f'callback>> \033[36m{token}\033[0m', end='')
    
    if token.endswith("。"):  # 日本語のピリオドに注意
        print("\nToken ends with a period. Generating audio...")
        result, output_filename = text_to_speech(buffered_text)
        buffered_text = ""  # バッファをクリア
        print(f"Generated audio file: {output_filename}")
        # 音声ファイルのURLを生成（後ほどクライアントに送信）
        audio_url = request.url_root + 'audio/' + output_filename
        send_audio_url_to_client(audio_url)


def get_chain(callback_streaming=None):
    callback_manager = CallbackManager([MyCustomCallbackHandler(callback_streaming)])
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", callback_manager=callback_manager, streaming=True)
    return llm

class MyCustomCallbackHandler(BaseCallbackHandler):
    def __init__(self, callback):
        self.callback = callback
    """Custom CallbackHandler."""
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.callback is not None:
            self.callback(token) 
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""

def openai_qa(query, callback_streaming=None):
    print(f"Received query for QA: {query}")
    llm = get_chain(callback_streaming)
    response = llm.invoke(query)
    print(f"LLM response: {response}")
    return response

@app.route('/text', methods=['POST'])
def handle_message():
    data = request.json  # JSON形式のデータを解析
    query = data.get('text')  # 'text'キーでテキストデータを取得
    if not query:
        return "No text provided", 400
    print(f"Handling synthesize_text event for text: {query}")
    response = openai_qa(query, dummy_callback)
    return json.dumps({"response": response}), 200

if __name__ == '__main__':
    app.run(debug=True)
