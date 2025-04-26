# Checkout-AI-fun
## Summary
The idea is to combine information sharing, AI learning and better marketing. The world is full of marketing that is not really connected to anything and it is coming to users at wrong time annoying them. This idea creates a community (emotional engagement), AI learning (how it predicts), marketing and product information.

A camera captures a store's chechout conveyor belt in real-time. Viewers can see additional information about the products, along with AI-powered predictions of what might appear next. The screen also displays statistics showing how AI is learning over time. At the same time, information about the heatlhiness is shared and the store and various brands can promote their products and services through the platform and viewers have possibility to do better choices when shopping. 

Watching is an option for doom-scrolling and for relaxing with endless TV series. However, viewers are still connected to real world - they see real shopping carts and can find emotional connection to the community. 

## Background
The solution is designed for entertainment, information sharing and marketing purposes. For instance, many people do not know how to select climate-friendly or healthy products. The target audience includes people who have enjoyed similar streams - such as Prisma checkout, Norwegian train routes, fireplace videos - and who either enjoy passive viewing or want to actively participate. Checkout AI Fun also offers unique and engaging conversion starters for social events, parties or casual gatherings.

The concept combines slow TV with artificial intelligence. There is something oddly relaxing about watching products move along a conveyer belt - as noted in this article [Livelähetys Prisman kassahihnalta nousi hitiksi](https://www.is.fi/viihde/art-2000005268226.html). At the same time, AI is a trending topic, yet still not widely understood. This project offers a chance to bring AI closer to a mainstream audience by visualizing how it "learns" from everyday behaviour - like the items we purchase.

The background information can tell facts about the products. For instance, "carrots are now at their best and very affordable. They are possible to add almos all meals." 

As a marketing tool, the platform can highlight product invormation, share discount codes and suggest recipe ideas. 

## How is it used?
Checkout AI Fun is abailable to users online and potentially also through the store's app. There is an interactive chat environment where viewers can participate in real time. Statistics of AI's predictions are displayed on the screen, visualizing how successful the AI model has been. Based on experiences from other popular live streams (Lintulautalive, Norppalive), many users keep the stream running throughout the day, even while working.

The development of Checkout AI Fun requires funding. Ideally, a sponsor - preferably the store itself - could support the initial launch, with later monetization through in-game purchases or user contributions.

Legal considerations - especially regarding GDPR - are a key part of the development process. Shoppers' data cannot be used without their consent, so clear signage is required at the checkout area. Shoppers are not filmed, but it might possible identify individuals by combining time, location and shopping cart content.

<img src="https://github.com/tiinasip/AI_idea/blob/main/Kuva_AI_Fun.jpg" width="200">
(Picture created with ChatGPT)

## Data sources and AI methods
### Description of the process
A live video stream continuously feeds frames to the AI model. A webcam captures the products from either a top-down or side angle. Then AI detects items and updates a dynamic item list in real time. Object detection is powered by pre-trained models - most likely YOLO, though alternatives like Detectron2, or OpenAI’s CLIP combined with image recognition are also viable.

When a new item appears on the conveyor belt, the AI predicts what might come next using a lightweight neural network. The entire system runs as a livestream, with a frontend display built using Streamlit, Flask or OBS, and a Python-based backend managing the AI processing.

### Selected techniques
YOLO (You Only Look Once) is a fast and accurate real-time object detection method, and it is ideal for live streams and low-latency apps. The implementations uses models YOLOv5 or YOLOv8, with the Ultralytics version preferred for its ease of integration and support.

For the prediction task, a small neural network is selected. Specifically, and LSTM (Long Short-Term Memory) network - a type of Recurrent Neural Network (RNN) - is used due to its ability to retain information across longer sequences. LSTM solves the short-memory limitation of standard RNNs, making it suitable for predicting the next time based on a sereies of previously detected products. However, LSTMs typically require a large amount of training data and are more computationally intensive than simpler models. 

An alternative considered was the Markov chain, which is a lightweight and easy-to-implement linear mathematical approach. However, since it only considers the immediate previous step, it cannot capture longer-term dpendencies eg. butter -> ham -> cheese), making it less effective for this use case.

### Statistics
Dashboard on the screen shows the output, prediction and prediction confidence history. 
1. Output is shown as text: "Apples!"
2. Prediction is shown as "Next likely item: milk" or in an enhanced version top 3 guesses with confidence such as milk - 65%, bread - 20% and butter - 5%.
3. Prediction confidence history bar shows, how confident the model has been over time.

For entertainment, there is also an Ai thinking buble commenting. "I've seen bread and butter, I suppose there could be cheese and ham as well.", "Now I am so confused, I just have to make a wild guess - eggs!" This buble would show also facts about nutrition.

Technically, Pygame is used to create the dashboard because it supports livestreams and is typically used in games.

## Code examples
### YOLO ultralytics
A simple code example for detecting objects in an image in Yolo ultralytics:
```
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolov8n.pt")  # 'n' = nano, super fast

# Run detection on an image
results = model("checkout.jpg")

# Show results
results[0].show()
````
### LSTM
A simple code example of predefined item associations. Dataset is prepared by creating shopping sequences and product names are tokenized into numerical sequences thtat the model can learn from:
```
["bread", "milk", "eggs", "cheese"]
["beer", "chips", "salsa"]
["yogurt", "granola", "banana"]
```
```
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_sequences)
```
Then LSTM is trained. Below you find a simplified example TensorFlow/Keras example (TensorFlow is open-source machine learning and deep learning framework by Google. Keras is API running on the top of TensorFlow. These are selected to create a prototype, as they are relatively easy to use and are also scalable to production use.)
```
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer

# Example training data
sequences = [["milk", "eggs", "cheese"], ["beer", "chips", "salsa"]]

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
encoded = tokenizer.texts_to_sequences(sequences)

# Create input-output pairs
X, y = [], []
for seq in encoded:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])

# Pad sequences
X = pad_sequences(X)
y = np.array(y)

# Define model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=X.shape[1]))
model.add(LSTM(50))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=1)
````
Predcition example code:
```
input_seq = ["milk", "eggs"]
encoded_input = tokenizer.texts_to_sequences([input_seq])
padded = pad_sequences(encoded_input, maxlen=X.shape[1])
predicted = model.predict(padded)
predicted_word = tokenizer.index_word[np.argmax(predicted)]
print(f"Predicted next item: {predicted_word}")
```

## Challenges
* GDPR and AI legislation must be reviewed before implementation, as privacy may become a concern.
* The chat feature needs moderation. 
* Technically, the project is feasible.
* Financially, it requires an investor, ideally a retailer, who could allocate their marketing budget for this initiative.
* Although one goal is to spread AI knowledge, most users are unlikely to gain significant insights from Checkout AI Fun. 

## What next?
It is easy to develope the idea further. Bonus features could be:
* Price estimation (“This belt looks like it’ll cost about €32.50”)
* Shopper profiling (“This looks like a cabin weekend stock-up”)
* Combinations ("Those items will end up to apple pie!")
* A commentator voice or chatbot (“Next up… yogurt? Suspense builds!”)
* More than prediction models competing against each other ("AI Alice is winning AI Joe today, and you viewers have lost to both of them!")
* Add self-checkout machines or other cashout conveyer belts and compare
* Viewers could make own suggestions for better forecast methods and also compete against AI with their own guesses.

Checkout viewer could become a community, as groceries and other everyday products are interesting and everyone has some emotions to them or brands. For instance
* "Who buys beer in the morning?"
* "You have money to buy a mango but then you buy the cheapest non-delphine friendly tuna!"
* "There is only meat products!"
* "There is only veggies, absolutely nothing to eat!"
* "Why do you buy 5 different sorts of cat food?"
* "Someone is going to bake, I guess it will be a chocolate cake!"

Items could extend beyond groceries - perhaps a large second-hand-store, a tech store, or even train services or roads could work. The scalability of the idea is strong.

If I decide to start the project, I would need technical and legal advice. There is a risk that no store would be willing to cooperate. However, if I secure an agreement with a store, I am confident the funding can be obtained. 

My technical skills are currently not at the required level, but finding a technical partner is feasible. I would also need assistance with marketing. 

## Acknowledgments
* ChatGPT 
* [Livelähetys Prisman kassahihnalta nousi hitiksi](https://www.is.fi/viihde/art-2000005268226.html)

Inspiration came also from
* [Lintulautalive](https://yle.fi/a/74-20146706)
* [Norppalive](https://wwf.fi/luontolive/norppalive/)
* [Slow tv](https://en.wikipedia.org/wiki/Slow_television)
* [An Interpretive Study on the Public's Preference for the Fireplace Videos](https://www.researchgate.net/publication/371228429_An_Interpretive_Study_on_the_Public's_Preference_for_the_Fireplace_Videos)
