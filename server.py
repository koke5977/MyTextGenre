import json
import flask
from flask import request
import mytext

TM_PORT_NO = 8085
app = flask.Flask(__name__)
print("http://localhost:" + str(TM_PORT_NO))

# テスト用
text = """
最近新しいスマートフォンを買いました。動画サイトで
有名な映画をたまに観賞しています。
"""
label, per, no = mytext.check_genre(text)
print(text, label, per, no)

@app.route('/', methods=['GET'])
def index():
    with open("index.html", "rb") as f:
        return f.read()

@app.route('/api', methods=['GET'])
def api():
    q = request.args.get('q', '')
    if q == '':
      return '{"label": "テキストに文章を入れてください", "per":0}'
    print("q=", q)
    
    label, per, no = mytext.check_genre(q)
    return json.dumps({
      "label": label, 
      "per": per,
      "genre_no": no
    })
    
if __name__ == '__main__':

    app.run(debug=False, port=TM_PORT_NO)

