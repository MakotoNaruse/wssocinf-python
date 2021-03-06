# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

from __future__ import unicode_literals

import sys
sys.path.append('./neural_net')

import datetime
import errno
import json
import os
import sys
import tempfile
from argparse import ArgumentParser
import urllib
from urllib import parse
from urllib import request
import requests
from flask import Flask, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from image_score import ImageScore


from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageAction,
    ButtonsTemplate, ImageCarouselTemplate, ImageCarouselColumn, URIAction,
    PostbackAction, DatetimePickerAction,
    CameraAction, CameraRollAction, LocationAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage, FileMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent,
    MemberJoinedEvent, MemberLeftEvent,
    FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, SpacerComponent, IconComponent, ButtonComponent,
    SeparatorComponent, QuickReply, QuickReplyButton,
    ImageSendMessage)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# create image scoreing class instance
image_score = ImageScore()

# function for create tmp dir for download content
def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'

#{'id': 1, 'message': 'Discovered user', 'status': 1}
def get_user_identity(user_id):
    user_id = user_id
    param = {
        'user_id':user_id
    }
    url = "https://wssocinf-5-web.herokuapp.com/api/get_user?"
    paramStr = urllib.parse.urlencode(param)
    r = requests.get(url + paramStr)
    json_response = r.content.decode()
    dict_json = json.loads(json_response)
    return dict_json

def change_situation(user_id, situation):
    user_id = user_id
    situation = situation
    param = {
        'user_id':user_id,
        'situation':situation
    }
    url = "https://wssocinf-5-web.herokuapp.com/api/change_situation?"
    paramStr = urllib.parse.urlencode(param)
    r = requests.post(url + paramStr)
    json_response = r.content.decode()
    dict_json = json.loads(json_response)

def get_recipe(recipe_id):
    recipe_id = recipe_id
    param = {
        'recipe_id':recipe_id
    }
    url = "https://wssocinf-5-web.herokuapp.com/api/get_recipe?"
    paramStr = urllib.parse.urlencode(param)
    r = requests.get(url + paramStr)
    json_response = r.content.decode()
    dict_json = json.loads(json_response)
    return(dict_json)

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    user_dict = get_user_identity(event.source.user_id)
    if user_dict['time'] > 10800:
        change_situation(event.source.user_id, 0)
        user_dict['situation'] = 0
    if text == 'getid':
        if isinstance(event.source, SourceUser):
            user_dict = get_user_identity(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='id: ' + str(user_dict['id'])),
                    TextSendMessage(text='situation: ' + str(user_dict['situation'])),
                    TextSendMessage(text='status: ' + str(user_dict['status'])),
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Cannot connect to the server"))
    elif user_dict['situation'] == 0:
        profile = line_bot_api.get_profile(event.source.user_id)
        change_situation(event.source.user_id, 1)
        temp_text = str(profile.display_name) + 'さん初めまして、私はお料理お姉さんよ。もしかして、今晩のメニューに悩んでいるんじゃない？'
        confirm_template = ConfirmTemplate(text=temp_text, actions=[
            MessageAction(label='はい', text='はい'),
            MessageAction(label='いいえ', text='いいえ'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif user_dict['situation'] == 1 and text == 'いいえ':
        change_situation(event.source.user_id, 0)
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="あら、そうなの。また料理に困ったら声をかけてね！"))
    elif user_dict['situation'] == 1 and text == 'はい':
        change_situation(event.source.user_id, 3)
        temp_text = 'じゃあ私があなたの気分からお料理を提案してあげるわ！今日はお肉の気分？'
        confirm_template = ConfirmTemplate(text=temp_text, actions=[
            MessageAction(label='はい', text='はい'),
            MessageAction(label='いいえ', text='いいえ'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif user_dict['situation'] == 3 and text == 'いいえ':
        change_situation(event.source.user_id, 4)
        temp_text = 'じゃあ海鮮の気分？'
        confirm_template = ConfirmTemplate(text=temp_text, actions=[
            MessageAction(label='はい', text='はい'),
            MessageAction(label='いいえ', text='いいえ'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif user_dict['situation'] == 3 and text == 'はい':
        change_situation(event.source.user_id, 5)
        temp_text = 'ちょっと凝った料理に挑戦してみる？'
        confirm_template = ConfirmTemplate(text=temp_text, actions=[
            MessageAction(label='はい', text='はい'),
            MessageAction(label='いいえ', text='いいえ'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif user_dict['situation'] == 4 and text == 'いいえ':
        change_situation(event.source.user_id, 12)
        temp_text = 'あなたは今日はベジタリアンなのね！じゃあこのお肉を使わない豆腐ハンバーグはどうかな？'
        recipe_dict = get_recipe(6)
        img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Tofu_Hamburg_steak_20141107.jpg/200px-Tofu_Hamburg_steak_20141107.jpg'
        buttons_template = ButtonsTemplate(
            thumbnail_image_url=img_url,
            title='豆腐ハンバーグ', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 4 and text == 'はい':
        change_situation(event.source.user_id, 13)
        temp_text = 'じゃあこの「海鮮アボカド」を作ってみない？'
        recipe_dict = get_recipe(7)
        img_url = recipe_dict['img_url']
        buttons_template = ButtonsTemplate(
            thumbnail_image_url=img_url,
            title='海鮮アボカド', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 5 and text == 'いいえ':
        change_situation(event.source.user_id, 6)
        temp_text = 'がっつりしたものが食べたいの？'
        confirm_template = ConfirmTemplate(text=temp_text, actions=[
            MessageAction(label='はい', text='はい'),
            MessageAction(label='いいえ', text='いいえ'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif user_dict['situation'] == 5 and text == 'はい':
        change_situation(event.source.user_id, 8)
        temp_text = 'じゃあこの「ビーフウェリントン」を作ってみない？　とっても豪華なイギリスの肉料理よ！'
        buttons_template = ButtonsTemplate(
            thumbnail_image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Beef_Wellington_-_Crosscut.jpg/800px-Beef_Wellington_-_Crosscut.jpg',
            title='ビーフウェリントン', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 6 and text == 'いいえ':
        change_situation(event.source.user_id, 9)
        temp_text = 'じゃあこの「タコライス」はいかが？'
        buttons_template = ButtonsTemplate(
            thumbnail_image_url='https://upload.wikimedia.org/wikipedia/commons/4/45/Taco_Rice1.JPG',
            title='タコライス', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 6 and text == 'はい':
        change_situation(event.source.user_id, 7)
        temp_text = 'じゃあこの「ビーフステーキ」がいいんじゃないかしら？'
        buttons_template = ButtonsTemplate(
            thumbnail_image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/150418_Awaji_beef_at_Sumoto_Hyogo_pref_Japan02s5.jpg/800px-150418_Awaji_beef_at_Sumoto_Hyogo_pref_Japan02s5.jpg',
            title='ビーフステーキ', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 7 and text == 'いいえ':
        change_situation(event.source.user_id, 0)
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="う〜ん、難しい子ねえ。また気が向いたら話しかけるのよ！"))
    elif user_dict['situation'] == 7 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(4)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="やっぱりステーキだよね！じゃあこのレシピにしたがって作ってみるのよ！できたら写真を送ってね〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 8 and text == 'いいえ':
        change_situation(event.source.user_id, 0)
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="あら、残念。じゃあまた今度ね！"))
    elif user_dict['situation'] == 8 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(2)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="やった〜！じゃあこのレシピにしたがって作ってみるのよ！できたら写真を送ってね〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 9 and text == 'いいえ':
        change_situation(event.source.user_id, 10)
        temp_text = 'それなら「青椒肉絲」ならどう？'
        buttons_template = ButtonsTemplate(
            thumbnail_image_url='https://upload.wikimedia.org/wikipedia/commons/9/9d/Pepper_steak.jpg',
            title='青椒肉絲', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 9 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(3)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="そうこなくっちゃ！じゃあこのレシピにしたがって作ってみるのよ！できたら写真を送ってね〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 10 and text == 'いいえ':
        change_situation(event.source.user_id, 0)
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="う〜ん、難しい子ねえ。また気が向いたら話しかけるのよ！"))
    elif user_dict['situation'] == 10 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(5)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="今日は中華に挑戦よ！じゃあこのレシピにしたがって作ってみるのよ！できたら写真を送ってね〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 12 and text == 'いいえ':
        change_situation(event.source.user_id, 14)
        temp_text = 'じゃあこのナスとチーズのベジタリアン料理でどうだ〜！！'
        recipe_dict = get_recipe(8)
        img_url = recipe_dict['img_url']
        buttons_template = ButtonsTemplate(
            thumbnail_image_url=img_url,
            title='ナスとチーズのベジタリアン', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 12 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(6)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="そうこなくっちゃ！　じゃあこのレシピにしたがって作ってみるのよ。完成したら写真を送ってね〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 13 and text == 'いいえ':
        change_situation(event.source.user_id, 15)
        temp_text = 'じゃあこの「ガーリックシュリンプ」ならどうだ！！'
        recipe_dict = get_recipe(9)
        img_url = recipe_dict['img_url']
        buttons_template = ButtonsTemplate(
            thumbnail_image_url=img_url,
            title='ガーリックシュリンプ', text=temp_text, actions=[
                MessageAction(label='はい', text='はい'),
                MessageAction(label='いいえ', text='いいえ'),
            ]
        )
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(
            event.reply_token, [
                template_message,
            ]
        )
    elif user_dict['situation'] == 13 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(7)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="そうこなくっちゃ！じゃあこのレシピにしたがって作ってみるのよ！完成したら写真を送ってね〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 14 and text == 'いいえ':
        change_situation(event.source.user_id, 0)
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="もう！　難しいんだから！　勝手にしなさい(≧ヘ≦　)ﾌﾟｲｯ!!"))
    elif user_dict['situation'] == 14 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(8)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="じゃあこのレシピにしたがって作ってね！完成したら写真を送るのよ〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 15 and text == 'いいえ':
        change_situation(event.source.user_id, 0)
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="もう！　難しいんだから！　勝手にしなさい(≧ヘ≦　)ﾌﾟｲｯ!!"))
    elif user_dict['situation'] == 15 and text == 'はい':
        change_situation(event.source.user_id, 11)
        recipe_dict = get_recipe(9)
        temp_text = recipe_dict['recipe_text']
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="じゃあこのレシピにしたがって作ってね！完成したら写真を送るのよ〜"),
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] == 11:
        temp_text = 'できたら写真を送ってね〜'
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text=temp_text),
            ]
        )
    elif user_dict['situation'] >= 1 and user_dict['situation'] <= 15:
        temp_text = '「はい」か「いいえ」で答えてね！'
        confirm_template = ConfirmTemplate(text=temp_text, actions=[
            MessageAction(label='はい', text='はい'),
            MessageAction(label='いいえ', text='いいえ'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif text == 'profile':
        if isinstance(event.source, SourceUser):
            profile = line_bot_api.get_profile(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='Display name: ' + profile.display_name),
                    TextSendMessage(text='Status message: ' + str(profile.status_message))
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't use profile API without user ID"))
    elif text == 'quota':
        quota = line_bot_api.get_message_quota()
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text='type: ' + quota.type),
                TextSendMessage(text='value: ' + str(quota.value))
            ]
        )
    elif text == 'quota_consumption':
        quota_consumption = line_bot_api.get_message_quota_consumption()
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text='total usage: ' + str(quota_consumption.total_usage)),
            ]
        )
    elif text == 'push':
        line_bot_api.push_message(
            event.source.user_id, [
                TextSendMessage(text='PUSH!'),
            ]
        )
    elif text == 'multicast':
        line_bot_api.multicast(
            [event.source.user_id], [
                TextSendMessage(text='THIS IS A MULTICAST MESSAGE'),
            ]
        )
    elif text == 'broadcast':
        line_bot_api.broadcast(
            [
                TextSendMessage(text='THIS IS A BROADCAST MESSAGE'),
            ]
        )
    elif text.startswith('broadcast '):  # broadcast 20190505
        date = text.split(' ')[1]
        print("Getting broadcast result: " + date)
        result = line_bot_api.get_message_delivery_broadcast(date)
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text='Number of sent broadcast messages: ' + date),
                TextSendMessage(text='status: ' + str(result.status)),
                TextSendMessage(text='success: ' + str(result.success)),
            ]
        )
    elif text == 'bye':
        if isinstance(event.source, SourceGroup):
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='Leaving group'))
            line_bot_api.leave_group(event.source.group_id)
        elif isinstance(event.source, SourceRoom):
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='Leaving group'))
            line_bot_api.leave_room(event.source.room_id)
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't leave from 1:1 chat"))
    elif text == 'image':
        url = request.url_root + '/static/logo.png'
        app.logger.info("url=" + url)
        line_bot_api.reply_message(
            event.reply_token,
            ImageSendMessage(url, url)
        )
    elif text == 'confirm':
        confirm_template = ConfirmTemplate(text='Do it?', actions=[
            MessageAction(label='Yes', text='Yes!'),
            MessageAction(label='No', text='No!'),
        ])
        template_message = TemplateSendMessage(
            alt_text='Confirm alt text', template=confirm_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif text == 'buttons':
        buttons_template = ButtonsTemplate(
            title='My buttons sample', text='Hello, my buttons', actions=[
                URIAction(label='Go to line.me', uri='https://line.me'),
                PostbackAction(label='ping', data='ping'),
                PostbackAction(label='ping with text', data='ping', text='ping'),
                MessageAction(label='Translate Rice', text='米')
            ])
        template_message = TemplateSendMessage(
            alt_text='Buttons alt text', template=buttons_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif text == 'carousel':
        carousel_template = CarouselTemplate(columns=[
            CarouselColumn(text='hoge1', title='fuga1', actions=[
                URIAction(label='Go to line.me', uri='https://line.me'),
                PostbackAction(label='ping', data='ping')
            ]),
            CarouselColumn(text='hoge2', title='fuga2', actions=[
                PostbackAction(label='ping with text', data='ping', text='ping'),
                MessageAction(label='Translate Rice', text='米')
            ]),
        ])
        template_message = TemplateSendMessage(
            alt_text='Carousel alt text', template=carousel_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif text == 'image_carousel':
        image_carousel_template = ImageCarouselTemplate(columns=[
            ImageCarouselColumn(image_url='https://via.placeholder.com/1024x1024',
                                action=DatetimePickerAction(label='datetime',
                                                            data='datetime_postback',
                                                            mode='datetime')),
            ImageCarouselColumn(image_url='https://via.placeholder.com/1024x1024',
                                action=DatetimePickerAction(label='date',
                                                            data='date_postback',
                                                            mode='date'))
        ])
        template_message = TemplateSendMessage(
            alt_text='ImageCarousel alt text', template=image_carousel_template)
        line_bot_api.reply_message(event.reply_token, template_message)
    elif text == 'imagemap':
        pass
    elif text == 'flex':
        bubble = BubbleContainer(
            direction='ltr',
            hero=ImageComponent(
                url='https://example.com/cafe.jpg',
                size='full',
                aspect_ratio='20:13',
                aspect_mode='cover',
                action=URIAction(uri='http://example.com', label='label')
            ),
            body=BoxComponent(
                layout='vertical',
                contents=[
                    # title
                    TextComponent(text='Brown Cafe', weight='bold', size='xl'),
                    # review
                    BoxComponent(
                        layout='baseline',
                        margin='md',
                        contents=[
                            IconComponent(size='sm', url='https://example.com/gold_star.png'),
                            IconComponent(size='sm', url='https://example.com/grey_star.png'),
                            IconComponent(size='sm', url='https://example.com/gold_star.png'),
                            IconComponent(size='sm', url='https://example.com/gold_star.png'),
                            IconComponent(size='sm', url='https://example.com/grey_star.png'),
                            TextComponent(text='4.0', size='sm', color='#999999', margin='md',
                                          flex=0)
                        ]
                    ),
                    # info
                    BoxComponent(
                        layout='vertical',
                        margin='lg',
                        spacing='sm',
                        contents=[
                            BoxComponent(
                                layout='baseline',
                                spacing='sm',
                                contents=[
                                    TextComponent(
                                        text='Place',
                                        color='#aaaaaa',
                                        size='sm',
                                        flex=1
                                    ),
                                    TextComponent(
                                        text='Shinjuku, Tokyo',
                                        wrap=True,
                                        color='#666666',
                                        size='sm',
                                        flex=5
                                    )
                                ],
                            ),
                            BoxComponent(
                                layout='baseline',
                                spacing='sm',
                                contents=[
                                    TextComponent(
                                        text='Time',
                                        color='#aaaaaa',
                                        size='sm',
                                        flex=1
                                    ),
                                    TextComponent(
                                        text="10:00 - 23:00",
                                        wrap=True,
                                        color='#666666',
                                        size='sm',
                                        flex=5,
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            footer=BoxComponent(
                layout='vertical',
                spacing='sm',
                contents=[
                    # callAction, separator, websiteAction
                    SpacerComponent(size='sm'),
                    # callAction
                    ButtonComponent(
                        style='link',
                        height='sm',
                        action=URIAction(label='CALL', uri='tel:000000'),
                    ),
                    # separator
                    SeparatorComponent(),
                    # websiteAction
                    ButtonComponent(
                        style='link',
                        height='sm',
                        action=URIAction(label='WEBSITE', uri="https://example.com")
                    )
                ]
            ),
        )
        message = FlexSendMessage(alt_text="hello", contents=bubble)
        line_bot_api.reply_message(
            event.reply_token,
            message
        )
    elif text == 'flex_update_1':
        bubble_string = """
        {
          "type": "bubble",
          "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
              {
                "type": "image",
                "url": "https://scdn.line-apps.com/n/channel_devcenter/img/flexsnapshot/clip/clip3.jpg",
                "position": "relative",
                "size": "full",
                "aspectMode": "cover",
                "aspectRatio": "1:1",
                "gravity": "center"
              },
              {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                  {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                      {
                        "type": "text",
                        "text": "Brown Hotel",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#ffffff"
                      },
                      {
                        "type": "box",
                        "layout": "baseline",
                        "margin": "md",
                        "contents": [
                          {
                            "type": "icon",
                            "size": "sm",
                            "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png"
                          },
                          {
                            "type": "icon",
                            "size": "sm",
                            "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png"
                          },
                          {
                            "type": "icon",
                            "size": "sm",
                            "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png"
                          },
                          {
                            "type": "icon",
                            "size": "sm",
                            "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png"
                          },
                          {
                            "type": "icon",
                            "size": "sm",
                            "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gray_star_28.png"
                          },
                          {
                            "type": "text",
                            "text": "4.0",
                            "size": "sm",
                            "color": "#d6d6d6",
                            "margin": "md",
                            "flex": 0
                          }
                        ]
                      }
                    ]
                  },
                  {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                      {
                        "type": "text",
                        "text": "¥62,000",
                        "color": "#a9a9a9",
                        "decoration": "line-through",
                        "align": "end"
                      },
                      {
                        "type": "text",
                        "text": "¥42,000",
                        "color": "#ebebeb",
                        "size": "xl",
                        "align": "end"
                      }
                    ]
                  }
                ],
                "position": "absolute",
                "offsetBottom": "0px",
                "offsetStart": "0px",
                "offsetEnd": "0px",
                "backgroundColor": "#00000099",
                "paddingAll": "20px"
              },
              {
                "type": "box",
                "layout": "vertical",
                "contents": [
                  {
                    "type": "text",
                    "text": "SALE",
                    "color": "#ffffff"
                  }
                ],
                "position": "absolute",
                "backgroundColor": "#ff2600",
                "cornerRadius": "20px",
                "paddingAll": "5px",
                "offsetTop": "10px",
                "offsetEnd": "10px",
                "paddingStart": "10px",
                "paddingEnd": "10px"
              }
            ],
            "paddingAll": "0px"
          }
        }
        """
        message = FlexSendMessage(alt_text="hello", contents=json.loads(bubble_string))
        line_bot_api.reply_message(
            event.reply_token,
            message
        )
    elif text == 'quick_reply':
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text='Quick reply',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=PostbackAction(label="label1", data="data1")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="label2", text="text2")
                        ),
                        QuickReplyButton(
                            action=DatetimePickerAction(label="label3",
                                                        data="data3",
                                                        mode="date")
                        ),
                        QuickReplyButton(
                            action=CameraAction(label="label4")
                        ),
                        QuickReplyButton(
                            action=CameraRollAction(label="label5")
                        ),
                        QuickReplyButton(
                            action=LocationAction(label="label6")
                        ),
                    ])))
    elif text == 'link_token' and isinstance(event.source, SourceUser):
        link_token_response = line_bot_api.issue_link_token(event.source.user_id)
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text='link_token: ' + link_token_response.link_token)
            ]
        )
    elif text == 'insight_message_delivery':
        today = datetime.date.today().strftime("%Y%m%d")
        response = line_bot_api.get_insight_message_delivery(today)
        if response.status == 'ready':
            messages = [
                TextSendMessage(text='broadcast: ' + str(response.broadcast)),
                TextSendMessage(text='targeting: ' + str(response.targeting)),
            ]
        else:
            messages = [TextSendMessage(text='status: ' + response.status)]
        line_bot_api.reply_message(event.reply_token, messages)
    elif text == 'insight_followers':
        today = datetime.date.today().strftime("%Y%m%d")
        response = line_bot_api.get_insight_followers(today)
        if response.status == 'ready':
            messages = [
                TextSendMessage(text='followers: ' + str(response.followers)),
                TextSendMessage(text='targetedReaches: ' + str(response.targeted_reaches)),
                TextSendMessage(text='blocks: ' + str(response.blocks)),
            ]
        else:
            messages = [TextSendMessage(text='status: ' + response.status)]
        line_bot_api.reply_message(event.reply_token, messages)
    elif text == 'insight_demographic':
        response = line_bot_api.get_insight_demographic()
        if response.available:
            messages = ["{gender}: {percentage}".format(gender=it.gender, percentage=it.percentage)
                        for it in response.genders]
        else:
            messages = [TextSendMessage(text='available: false')]
        line_bot_api.reply_message(event.reply_token, messages)
    else:
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=event.message.text))


@handler.add(MessageEvent, message=LocationMessage)
def handle_location_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        LocationSendMessage(
            title='Location', address=event.message.address,
            latitude=event.message.latitude, longitude=event.message.longitude
        )
    )


@handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        StickerSendMessage(
            package_id=event.message.package_id,
            sticker_id=event.message.sticker_id)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    ext = 'jpg'
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.' + ext
    dist_name = os.path.basename(dist_path)
    os.rename(tempfile_path, dist_path)
    # Image Score
    change_situation(event.source.user_id, 0)
    score = image_score.predict_score(dist_path)

    if score < 60:
        text = 'う〜ん、これは{}点ね…\n次はもうちょっと高得点を出せるように頑張ろう！'.format(score)
    elif 60 <= score and score < 80:
        text = '{}点よ。なかなかやるじゃない！\n次はもっと高得点を目指して頑張ろう！'.format(score)
    elif 80 <= score and score < 95:
        text = '{}点よ！素晴らしい出来だわ〜\nこの調子でお料理上手を目指しましょう！'.format(score)
    else:
        text = 'すっごーい！{}点！\nこれ以上ない素晴らしい出来だわ！'.format(score)

    line_bot_api.reply_message(
        event.reply_token, [
            TextSendMessage(text=text),
        ])

# Other Message Type
@handler.add(MessageEvent, message=(VideoMessage, AudioMessage))
def handle_content_message(event):

    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.' + ext
    dist_name = os.path.basename(dist_path)
    os.rename(tempfile_path, dist_path)


@handler.add(MessageEvent, message=FileMessage)
def handle_file_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix='file-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '-' + event.message.file_name
    dist_name = os.path.basename(dist_path)
    os.rename(tempfile_path, dist_path)

    line_bot_api.reply_message(
        event.reply_token, [
            TextSendMessage(text='Save file.'),
            TextSendMessage(text=request.host_url + os.path.join('static', 'tmp', dist_name))
        ])


@handler.add(UnfollowEvent)
def handle_unfollow(event):
    app.logger.info("Got Unfollow event:" + event.source.user_id)


@handler.add(JoinEvent)
def handle_join(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='Joined this ' + event.source.type))


@handler.add(LeaveEvent)
def handle_leave():
    app.logger.info("Got leave event")


@handler.add(PostbackEvent)
def handle_postback(event):
    if event.postback.data == 'ping':
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text='pong'))
    elif event.postback.data == 'datetime_postback':
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=event.postback.params['datetime']))
    elif event.postback.data == 'date_postback':
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=event.postback.params['date']))


@handler.add(BeaconEvent)
def handle_beacon(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text='Got beacon event. hwid={}, device_message(hex string)={}'.format(
                event.beacon.hwid, event.beacon.dm)))


@handler.add(MemberJoinedEvent)
def handle_member_joined(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text='Got memberJoined event. event={}'.format(
                event)))


@handler.add(MemberLeftEvent)
def handle_member_left(event):
    app.logger.info("Got memberLeft event")


@app.route('/static/<path:path>')
def send_static_content(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', type=int, default=8000, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    # create tmp dir for download content
    make_static_tmp_dir()

    app.run(debug=options.debug, port=options.port)
