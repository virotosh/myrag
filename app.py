'''
  Simple Flask app
'''
import json
import os
import flask

# Flask app object
app = flask.Flask(__name__)

# Routes
@app.route("/", methods=['GET'])
def home():
    '''
      Hello page
    '''
    return "<p>Hello, World!</p>"

# Entry function
def main():
    '''
      Main entry function
    '''

    app.run(port=8080,
            host='0.0.0.0')

if __name__ == "__main__":
    main()
