const path = require('path');
const webpack = require('webpack');

module.exports = {
    mode: 'production',
    entry: {
        server: './index.js',
    },
    output: {
        path: path.join(__dirname, 'build'),
        filename: 'index.js'
    },
    module: {
        rules: [
            {
                exclude: /node_modules/
            }
        ]
    }
};