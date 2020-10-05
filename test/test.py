from table_bert import TableBertModel
from table_bert import Table, Column

import torch

model = TableBertModel.from_pretrained(
    '/Users/mac/Desktop/syt/Deep-Learning/Repos/TaBERT/pretrained-models/tabert_base_k3/model.bin',
)

table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)

table2 = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453'),
        Column('Continent', 'text', sample_value='North America')
    ],
    data=[
        ['United States', '21,439,453', 'North America'],
        ['China', '27,308,857', 'Asia'],
        ['European Union', '22,774,165', 'Europe'],
    ]
).tokenize(model.tokenizer)

# To visualize table in an IPython notebook:
# display(table.to_data_frame(), detokenize=True)

context = 'show jhgdygfug countries ranked by GDP'
context_tokenized = model.tokenizer.tokenize(context)

context2 = 'show asian countries ranked by GDP'
context2_tokenized = model.tokenizer.tokenize(context2)

# model takes batched, tokenized inputs
with torch.no_grad():
    context_encoding, column_encoding, info_dict = model.encode(
        contexts=[context_tokenized, context2_tokenized],
        tables=[table, table2]
    )

print(context_tokenized)
print(table)
print(context2_tokenized)
print(table2)
print()
print(context_encoding.shape)
print(column_encoding.shape)
print()
print(context_encoding[1, -1])
print(column_encoding[0, -1])



