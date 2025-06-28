from flask import Flask, jsonify, request
from flask_cors import CORS
import Connect_to_DBPedia as ctd
import Calculate_user_genre as cug
import Calculate_user_best_actor as cuba

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

user = 1

@app.route('/api/getTopTen', methods=['GET'])
def get_top_ten():
    global user
    return jsonify(ctd.top_ten(user))

@app.route('/api/similar_movies', methods=['GET'])
def similar_movies():
    global user
    return jsonify(ctd.similar_movies(user))

@app.route('/api/getScopriDiPiu', methods=['GET'])
def get_scopri_di_piu():
    global user
    return jsonify(ctd.scopri_di_piu(user))

@app.route('/api/getTopTenByGenre', methods=['GET'])
def get_top_ten_by_genre():
    global user
    return jsonify(cug.get_top_film_genre(user))

@app.route('/api/getTopTenByActor', methods=['GET'])
def get_top_ten_by_actor():
    global user
    return jsonify(cuba.get_films_by_actor(user))

@app.route('/api/getUserInfo', methods=['GET'])
def get_user_info():
    global user
    return jsonify(ctd.get_user_information(user))

@app.route('/api/setUserId', methods=['POST'])
def set_user_id():
    global user
    data = request.get_json()
    user = int(data.get('userId', user))
    print("Post:", data.get('userId', user))
    return jsonify({"message": "userId updated", "userId": user})

@app.route('/api/getActorInformation', methods=['POST'])
def get_actor_info():
    global user
    data = request.get_json()
    actor = data.get('actor')
    print("Post:", data.get('userId', user))
    return jsonify(cuba.get_actor_info(user, actor))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
