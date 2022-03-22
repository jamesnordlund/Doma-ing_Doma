# Doma-ing Doma

[**Doma**](https://www.doma.com/about/).  Proper noun.  Title insurance and settlement services disruptor.

**Doma-ing**.  Verb meaning "to mimic the analysis in Doma's blog posts."  Neologisms are one way to ensure that NLP problems remain fun :upside_down_face:.

## Sources

Text information is compiled from:
1. [Doma'a press releases](https://www.doma.com/press/press-releases/)
2. [Doma's published resources for lenders](https://www.doma.com/resources-for-lenders/)
3. [Doma's blog posts](https://www.doma.com/category/writing-on-the-wall/)

It never hurts to throw up a word cloud and verify that the corpus generates sensible output.
![word cloud](corpus_word_cloud.png)

## Motivation and Results

Nice blog posts from Doma [like this one](https://www.doma.com/understanding-berts-semantic-interpretations/) show off how cool it can be to do a little visualization with models to learn more about them.

In particular, [this post](https://www.doma.com/neural-language-models-as-domain-specific-knowledge-bases/) highlights the need for tuning language models to a domain's corpus.

As an extension to the examples shown in that post, I'll compare standard BERT and ROBERTA language models (`bert-base-uncased` and `roberta-base` in the HuggingFace Hub) against language models trained on text published by Doma.
Here, the corpus is obviously comprised of materials that Doma uses to advertise itself.
Hence, a language model trained on this data should adapt to normative claims made by the company and be able to replicate that "Doma is awesome" attitude.

A few example strings are test:

| String | BERT | BERT Refined to Doma Corpus | ROBERTA | ROBERTA Refined to Doma Corpus |
| ------ | ---- | --------------------------- | ------- | ------------------------------ |
| "Doma wants the closing process to be \<mask\>." | ["complete", "completed"] | ["complete", "streamlined"] | ["transparent", "completed"] | ["seamless", "efficient"] |
| "Our approach to titling is \<mask\> for home buyers." | ["recommended", "suitable"] | ["essential", "revolutionary"] | ["ideal", "good"] | ["revolutionary", "empowering"] |
| "The traditional mortgage process is too \<mask\>." | ["complex", "expensive"] | ["complex", "costly"] | ["complicated", "complex"] | ["complex", "cumbersome"] |
| "With the aid of \<mask\>, we make closing faster." | ["water", "light"] | ["technology", "tech"] | ["gravity", "levers"] | ["technology", "science"] |

The last one is pretty funny.  In all instances, the model refined to Doma's corpus of marketing materials is better able to fill in sentences such that the text reads like something you would find on their website.
This is not surprising, a good model should be able to adapt to the domain of the language that it's shown.
However, the implications are cool.  Suppose you want to build a chat bot.  The corpus of company-published materials would definitely give the bot a personality that fits the company's vision.
