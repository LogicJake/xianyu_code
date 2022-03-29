from time import process_time_ns
import docx
from numpy import record
import pandas as pd


def is_bold(para):
    for c in para.runs:
        if c.bold:
            return True
    return False


def read_docx(path):
    records = {'科': [], '科_en': [], '属': [], '属_en': [], '种': [], '种_en': []}

    ke = None
    shu = None

    data = docx.Document(path)
    for para in data.paragraphs:
        style = para.style.name
        text = para.text

        if text == '' or len(text) < 2:
            continue

        if style == 'Heading 2':
            ke = text

        if style == 'Normal' and is_bold(para):
            shu = text

        if style == 'Normal' and not is_bold(
                para
        ) and ke is not None and shu is not None and '分布' not in text:
            space_pos = ke.rfind(' ')
            records['科'].append(ke[space_pos + 1:])
            records['科_en'].append(ke[:space_pos])

            space_pos = shu.rfind(' ')
            records['属'].append(shu[space_pos + 1:])
            records['属_en'].append(shu[:space_pos])

            space_pos = text.rfind(' ')
            records['种'].append(text[space_pos + 1:])
            records['种_en'].append(text[:space_pos])

    df = pd.DataFrame(records)
    df.to_excel('1.xlsx', index=False)


if __name__ == "__main__":
    read_docx("./青海祁连山区种子植物名录(1).docx")