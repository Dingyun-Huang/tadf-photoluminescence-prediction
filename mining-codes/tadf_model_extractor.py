import importlib.resources
import json
import logging
import os
import re

from chemdataextractor.doc import (Caption, Citation, Document, Heading,
                                   Paragraph, Text)
from chemdataextractor.model import ThemeCompound
from chemdataextractor.reader.markup import HtmlReader
from chemdataextractor.text import get_encoding
from lxml import etree
from lxml.html import HTMLParser

logger = logging.getLogger("TADFExtractor")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("tadf_extractor.log", mode="a")
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define log format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

tadf_blocklist = (
    importlib.resources.files("tadf_models")
    .joinpath("tadf_blocklist_6_more_abbrev_enriched")
    .read_text(encoding="utf8")
)
element_blocklist = (
    importlib.resources.files("tadf_models")
    .joinpath("elements")
    .read_text(encoding="utf8")
)
tadf_blocklist = tadf_blocklist.split("\n")
element_blocklist = element_blocklist.split("\n")
ThemeCompound.name_blocklist = (
    ThemeCompound.name_blocklist + tadf_blocklist + element_blocklist
)

# Basic LaTeX symbol mappings
latex_symbols = {
    r'\alpha': 'α',
    r'\beta': 'β',
    r'\gamma': 'γ',
    r'\delta': 'δ',
    r'\epsilon': 'ε',
    r'\zeta': 'ζ',
    r'\eta': 'η',
    r'\theta': 'θ',
    r'\iota': 'ι',
    r'\kappa': 'κ',
    r'\lambda': 'λ',
    r'\mu': 'μ',
    r'\nu': 'ν',
    r'\xi': 'ξ',
    r'\pi': 'π',
    r'\rho': 'ρ',
    r'\sigma': 'σ',
    r'\tau': 'τ',
    r'\upsilon': 'υ',
    r'\phi': 'φ',
    r'\chi': 'χ',
    r'\psi': 'ψ',
    r'\omega': 'ω',
    r'\Gamma': 'Γ',
    r'\Delta': 'Δ',
    r'\Theta': 'Θ',
    r'\Lambda': 'Λ',
    r'\Xi': 'Ξ',
    r'\Pi': 'Π',
    r'\Sigma': 'Σ',
    r'\Phi': 'Φ',
    r'\Psi': 'Ψ',
    r'\Omega': 'Ω',
    r'\times': '×',
    r'\cdot': '-',
    r'\pm': '±',
    r'\leq': '≤',
    r'\geq': '≥',
    r'\neq': '≠',
    r'\approx': '≈',
    r'\infty': '∞',
    r'\rightarrow': '→',
    r'\leftarrow': '←',
    r'\Rightarrow': '⇒',
    r'\Leftarrow': '⇐',
    r'\leftrightarrow': '↔',
    r'\sum': '∑',
    r'\prod': '∏',
    r'\int': '∫',
    r'\partial': '∂',
    r'\nabla': '∇',
    r'\%': '%',
    r'\prime': "'",
}


hyphen_regex = "[-–—−]"

def convert_latex_to_plain(text):
    # Replace math environments with contents
    # text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)       # \[...\]
    # text = re.sub(r'\\begin\{.*?\}(.*?)\\end\{.*?\}', r'\1', text, flags=re.DOTALL)

    # Replace known LaTeX symbols
    for latex, symbol in latex_symbols.items():
        text = text.replace(latex, symbol)
    
    text = re.sub(r'\{([^}]*)\}', lambda m: '{' + m.group(1).replace(' ', '').replace('.', '-') + '}', text)
    text = re.sub(r'\$\$([^\$]*)\$\$', lambda m: '$$' + m.group(1).replace(' ', '').replace('.', '-') + '$$', text)   # $$...$$
    text = re.sub(r'\$([^\$]*)\$', lambda m: '$' + m.group(1).replace(' ', '').replace('.', '-') + '$', text)      # $...$

    # Remove remaining LaTeX commands like \frac{}, \left, \right, etc.
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)  # $$...$$
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)      # $...$
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\{|\}', '', text)
    text = re.sub(r"~", ' ', text)  # Remove tildes, carets, and underscores
    text = re.sub(r"[\^_]", '', text)
    text = re.sub(r"((\\\:)|(\\,)|(\\;)|(\\quad)|(\\quad)|(\\\!))+", " ", text)
    text = re.sub(r'\[(\d{1,3})([\,\-\–]\d{1,3})?\]', '', text)

    # Remove spaces around hyphens
    text = re.sub(r"\s?" + hyphen_regex + r"\s?", "-", text)
    # Remove extra hyphens
    text = re.sub(hyphen_regex + hyphen_regex +  r"+", "-", text)
    
    # Remove extra spaces
    text = re.sub(r'\s\s+', ' ', text)
    return text.strip()


def load_wiley_md(file_path):
    """
    Load a Wiley MD file and return a Document object.
    :param file_path: Path to the Wiley MD file.
    :return: Document object.
    """
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
    with open(file_path, 'r', encoding="utf-8") as file:
        i = 0
        for line in file.readlines():
            lines[i] = line
            i += 1

    elements = []
    for i, line in lines.items():
        line = line.replace(r"\$", "$")
        line = convert_latex_to_plain(line)
        lines[i] = line
        if line.startswith("#"):
            # print("Heading Line")
            if any([p in line.lower() for p in ending_phrases]):
                break
            elements.append(Heading(line.strip("#\n ")))
        elif line.startswith("<html>"):
            # print("Table Line")
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
    return Document(*elements)


class TADFExtractor:

    def __init__(self, paper_root, save_root, save_filename, target_paper, models):
        self.dic = None
        self.save_filename = save_filename
        self.paper_root = paper_root
        self.models = list(models)
        self.count = 0
        self.save_root = save_root
        self.target_paper = target_paper

    def is_incomplete_paper(self, fstring):
        logger.info("Checking completeness.")
        if (
            "(Note: The full text of this document is currently only available in the"
            in fstring
        ):
            # RSC case
            return True
        elif "<xocs:rawtext" in fstring:
            # Elsevier case
            return True
        return False

    def load_document(self, file):

        if file.endswith("ml"):
            d = Document.from_file(file)
            incomplete = False
            with open(file, encoding="utf8") as f:
                fstring = f.read()
            if self.is_incomplete_paper(fstring):
                incomplete = True
        elif file.endswith("md"):
            d = load_wiley_md(file)
            incomplete = False
        else:
            logger.error("File type not supported.")
            return None, True
        
        return d, incomplete

    def write_to_json(self):

        with open(
            f"{self.save_root}/{self.save_filename}.json",
            "a",
            newline="",
            encoding="utf8",
        ) as json_file:
            json.dump(self.dic, json_file, ensure_ascii=False)
            json_file.write("\n")

    def is_tadf(self, doc):
        """
        Method that check if a document is a TADF paper.
        :param chemdataextractor.doc.Document doc: Document object.
        :returns: if the document is a TADF paper or not.
        :rtype: Boolean.
        """
        delayed_exp = re.compile(r"[Dd]elayed")
        count = 0
        for element in doc.elements:
            if isinstance(element, Text):
                text = element.text
                count += len(delayed_exp.findall(text))
            if count >= 3:
                return True
        return False
    
    def save_footnotes(self):

        file_path = os.path.join(self.paper_root, self.target_paper)

        doc, incomplete = self.load_document(file_path)
        if incomplete:
            return
        if self.is_tadf(doc):
            pass
        else:
            return
        try:
            doi = doc.metadata.doi
        except IndexError:
            logger.info("MetaData is Empty.")
            doi = "https://doi.org/" + self.target_paper.replace("_", "/").replace(
                ".xml", ""
            ).replace(".html", "")
        doc.skip_elements = [Citation]
        


    def extraction(self):

        logger.info("Attempting to extract %s.", self.target_paper)

        file_path = os.path.join(self.paper_root, self.target_paper)

        doc, incomplete = self.load_document(file_path)
        if incomplete:
            logger.info("%s is incomplete!", self.target_paper)
            return
        if self.is_tadf(doc):
            logger.info("%s is a likely TADF paper.", self.target_paper)
        else:
            logger.info("%s is not likely a TADF paper.", self.target_paper)
            return
        logger.info("%s is complete, extracting.", self.target_paper)
        try:
            doi = doc.metadata.doi
        except IndexError:
            logger.info("MetaData is Empty.")
            doi = "https://doi.org/" + self.target_paper.replace("_", "/").replace(
                ".xml", ""
            ).replace(".html", "").replace(".md", "")
        doc.skip_elements = [Citation]

        doc.models = self.models
        rough = doc.records
        for r in rough:
            if isinstance(r, ThemeCompound):
                logger.debug("%s ThemeCompound skipped for saveing.", r)
                continue
            self.dic = r.serialize()
            self.dic["doi"] = doi
            self.dic["record_method"] = r.record_method
            try:
                self.dic["context"] = r.context
            except AttributeError:
                logger.warning("No context found for record %s.", r.serialize())
                self.dic["context"] = ""
            self.write_to_json()
            self.count += 1
        logger.info("%d records in %s.", self.count, self.target_paper)
        logger.info("%s extracted.", self.target_paper)
