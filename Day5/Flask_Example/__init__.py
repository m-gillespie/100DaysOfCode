from flask import Flask
app = Flask(__name__)

import Flask_Example.views.display
import Flask_Example.views.api


if __name__ == '__main__':
    app.run(debug=True)