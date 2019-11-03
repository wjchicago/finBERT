from flask import Flask, request
from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained("models/classifier_model/finbert-sentiment",num_labels=3,cache_dir=None)

@app.route("/sentiment")
def members():
    text = request.args.get('text')
    print(text)

    result = predict(text,model,write_to_csv=False,path=None)
    print(result)
    avg_score = result['sentiment_score'].mean()
    print(avg_score)
    if avg_score>0:
        return "positive"
    return "negative"


if __name__ == "__main__":
    app.run()



