from chemdataextractor.doc import Document, Paragraph, Heading, Caption
from chemdataextractor.reader.markup import HtmlReader
from lxml.html import HTMLParser
from lxml import etree
from chemdataextractor.text import get_encoding
from tadf_models.models import PhotoluminescenceWavelength

def read_elements(filepath):
    table_reader = HtmlReader()
    table_reader.table_body_row_css = "tr"
    ending_phrases = ["reference",
                      "acknowledgement",
                      "appendix",
                      "supporting information",
                      "citation",
                      "supplementary information",
                      "supplementary material",
                      "conflict of interest",
                      ]
    
    lines = {}
    with open(filepath, 'r') as file:
        i = 0
        for line in file.readlines():
            lines[i] = line
            i += 1

    elements = []
    for i, line in lines.items():
        if line.startswith("#"):
            # print("Heading Line")
            if any([p in line.lower() for p in ending_phrases]):
                break
            elements.append(Heading(line.strip("\n ")))
        elif line.startswith("<html>"):
            print("Table Line")
            caption = Caption(lines[i-3].strip("\n "))
            table_etree = etree.fromstring(line, parser=HTMLParser(encoding=get_encoding(line)))
            table_etree = table_etree.xpath("//table")[0]
            #print("Table etree")
            #print(etree.tostring(table_etree))
            table = table_reader._parse_table(table_etree, {}, {})[0]
            table.caption = caption
            elements.append(table)
        elif len(line.strip("\n ")) > 3:
            # print("Paragraph Line")
            elements.append(Paragraph(line.strip("\n ")))
    return elements


if __name__ == "__main__":
    elements = read_elements("/path/to/your/file.md")
    doc = Document(*elements)
    doc.models = [PhotoluminescenceWavelength]
    doc.records.serialize()
