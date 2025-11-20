'''
  Simple Flask app
'''
import json
import os
import flask

# Flask app object
app = flask.Flask(__name__,
                  static_url_path='/static',
                  static_folder='/static')

# Routes
@app.route("/", methods=['GET'])
def home():
    '''
      Hello page
    '''
    return 'Hello page'

# Entry function
def main():
    '''
      Main entry function
    '''

    app.run(debug=config["debug"],
            port=8080,
            host='0.0.0.0')

if __name__ == "__main__":
    main()
