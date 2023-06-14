# VIS2_WordStream

This project presents a visual representation of the motions put forth by different political
parties in the Austrian parliament. We utilize WordStream to depict the topics that each party
aimed to discuss over time.

Install the requirements with:

``
    pip install -r requirements.txt
``

To generate the HTML file in the html directory run:

``
    python -m src.plot
``

To generate the documentation run:

``
    pdoc src/ -o doc/
``

To generate the data for each legislative period run:

``
    cd src/datapreparation
    python api_reader.py
    python topic_transformation.py
``

Note that which legislative periods to download and transform are hard coded. The output is written to the data directory.

The font directory contains the fonts needed to generate the positions but is not needed for displaying the output.
