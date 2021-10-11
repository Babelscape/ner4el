![logo](./img/logo_ner4el.png)
--------------------------------------------------------------------------------

Code and resources for the paper [Named Entity Recognition for Entity Linking: What Works and What's Next]().

```bibtex
@inproceedings{tedeschi-etal-2021-ner4el,
    title = {{N}amed {E}ntity {R}ecognition for {E}ntity {L}inking: {W}hat Works and What's Next,
    author = "Tedeschi, Simone and Conia, Simone and Cecconi, Francesco and Navigli, Roberto",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic"

```

**Please consider citing our work if you use resources and/or code from this repository.**

In this work we focus on **Entity Linking (EL)**, a key task in NLP which aims at associating an ambiguous textual mention with a named entity in a knowledge base. It is a very **knowledge-intensive task** and current EL approaches requires massive amounts of training data – often millions of labeled items – in order to perform at their best, making the development of a high-performance EL system viable only to a **limited audience**. Hence, we study whether it is possible to **narrow the performance gap** between systems trained on limited (i.e., less than 20K labeled samples) and large amounts of data (i.e., millions of training samples). In particular, we take a look at **Named Entity Recognition (NER)** – the task of identifying specific words as belonging to predefined semantic types such as Person, Location, Organization – and how this task can be exploited to **improve a strong Entity Linking baseline in low-resource settings** without requiring any additional data. We show how and to what extent an EL  system can benefit from NER to enhance its entity  representations, improve candidate selection, select more  effective negative samples and enforce hard and soft  constraints on its output entities.

<br>

# Fine-Grained Classes for NER
In its standard formulation, NER distinguishes between four classes of entities: Person (PER), Location (LOC), Organization (ORG), and Miscellaneous (MISC).
Although NER systems that use these four classes have been found to be beneficial in downstream tasks, we argue that they might be too coarse-grained and, at the same time, not provide a sufficiently exhaustive coverage to also benefit EL, as many different entities would fall within the same Misc class.

For these reasons, **we introduce a new set of 18 finer-grained NER classes**, namely, Person (PER), Location (LOC), Organization (ORG), Animal (ANIM), Biology (BIO), Celestial Body (CEL), Disease (DIS), Event (EVE), Food (FOOD), Instrument (INST), Media (MEDIA), Monetary (MON), Number (NUM), Physical Phenomenon (PHYS), Plant (PLANT), Supernatural (SUPER), Time (TIME) and Vehicle (VEHI).

In order to use the newly introduced NER classes, we **automatically** label each Wikipedia entity with one of them by taking advantage of [WordNet](https://wordnet.princeton.edu/) and [BabelNet](https://babelnet.org/).

You can **DOWNLOAD** the resulting mapping here: [Wikipedia2NER-mapping](https://drive.google.com/file/d/1tnyYe1alAPP2L866bUq4MtUh687z7oE4/view?usp=sharing) (158MB).

The following table reports the statistics about number of articles for each of the 18 NER classes.

<div align="center">

| NER Class | Number of Wikipedia Articles |
| :------------- | -------------: |
| Person | 1,886K|
| Organization | 439K|
| Location | 1,228K|
| Animal | 330K|
| Biology | 16K|
| Celestial Body | 13K|
| Disease | 9K|
| Event | 249K|
| Food | 15K|
| Instrument | 52K|
| Media | 703K|
| Monetary | 2K|
| Number | 1K|
| Physical Phenomenon | 2K|
| Plant | 51K|
| Supernatural | 6K|
| Time | 9K|
| Vehichle | 78K|

</div>

<br>

# Other Resources
Here you can download other resources needed to run the code, but also useful for other purposes (e.g., as a starting point for other EL projects).

<center>

| Resource | Description |
| ------------- | :------------- |
| [Alias Table](https://drive.google.com/file/d/13iro8M2KVONWANcgna_3zxxPZl9b7TVC/view?usp=sharing) (732MB) | A dictionary that associates each textual mention with a set of possible candidates <br>(i.e., a set of possible Wikipedia IDs)|
| [Descriptions Dictionary](https://drive.google.com/file/d/1kv1yxbrqvNgONcjuu2XNaoDrs6acOs4t/view?usp=sharing) (2.7GB) | A dictionary that associates each Wikipedia ID with its textual description|
| [Counts Dictionary](https://drive.google.com/file/d/1uKAO2866GAwVYdq1Rda6v-C2TZvoWOoZ/view?usp=sharing) (222MB) | A dictionary that associates each Wikipedia ID with its frequency in Wikipedia <br>(i.e., the sum of all the wikilinks that refer to that page)|
| [Titles Dictionary](https://drive.google.com/file/d/1hoUfhfNTP_73mcrYoWVBrwHQ8RXP2OSY/view?usp=sharing) (178MB) | A dictionary that associates the title of a Wikipedia page with its corresponding Wikipedia ID|
| [NER Classifier](https://drive.google.com/file/d/1hYrSfuogz0tdvhY9UA0bxgNPWUBJAsjC/view?usp=sharing) (418MB) | The pretrained NER classifier used for the NER-constrained decoding and NER-enhanced candidate generation contributions|

</center>

<br>

# Data
The only training data that we use for our experiments are the training instances from the **AIDA-YAGO-CoNLL** training set. We evaluate our systems on the **validation** split of **AIDA-YAGO-CoNLL**.
For **testing** we use the test split of **AIDA-YAGO-CoNLL**, and the **MSNBC**, **AQUAINT**, **ACE2004**, **WNED-CWEB** and **WNED-WIKI** test sets.

We preprocessed the datasets and converted them in the format:
<br>
<center>
{"mention": MENTION, "left_context": LEFT_CTX, "right_context": RIGHT_CTX"}
</center>
<br>
<br>

The preprocessed datasets are already available in this repository:
- [AIDA-YAGO-CoNLL (Train)](./ner4el/data/aida_train.jsonl)
- [AIDA-YAGO-CoNLL (Dev)](./ner4el/data/aida_dev.jsonl)
- [AIDA-YAGO-CoNLL (Test)](./ner4el/data/aida_test.jsonl)
- [MSNBC](./ner4el/data/msnbc_test.jsonl)
- [AQUAINT](./ner4el/data/aquaint_test.jsonl)
- [ACE2004](./ner4el/data/ace2004_test.jsonl)
- [WNED-CWEB](./ner4el/data/cweb_test.jsonl)
- [WNED-WIKI](./ner4el/data/wiki_test.jsonl)

<br>

# License 
NER4EL is licensed under the CC BY-SA-NC 4.0 license. The text of the license can be found [here](https://github.com/Babelscape/wikineural/blob/master/LICENSE). The code in this repository is built on [![](https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/lucmos/nn-template).