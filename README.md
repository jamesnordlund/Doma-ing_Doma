# Doma-ing Doma

[**Doma**](https://www.doma.com/about/).  Proper noun.  Title insurance and settlement services disruptor.

**Doma-ing**.  Verb meaning "to mimic the analysis in Doma's blog posts."  Neologisms are one way to ensure that NLP problems remain fun :upside_down_face:.

## Sources

Text information is compiled from:
1. [Doma'a press releases](https://www.doma.com/press/press-releases/)
2. [Doma's published resources for lenders](https://www.doma.com/resources-for-lenders/)
3. [Doma's log posts](https://www.doma.com/category/writing-on-the-wall/)

It never hurts to throw up a word cloud and verify that the corpus generates sensible output.
![word cloud](corpus_word_cloud.png)

## Motivations

Nice blog posts from Doma [like this one](https://www.doma.com/understanding-berts-semantic-interpretations/) show off how cool it can be to do a little visualization with models to learn more about them.

In particular, [this post](https://www.doma.com/neural-language-models-as-domain-specific-knowledge-bases/) highlights the need for tuning language models to a domain's corpus.

As an extension to the examples shown in that post, I'll compare a standard BERT language model (`bert-base-uncased` in the HuggingFace Hub) against a language model trained on text published by Doma.
Here, the corpus is obviously comprised of materials that Doma uses to advertise itself.
Hence, a language model trained on this data should adapt to normative claims made by the company and be able to replicate that "Doma is awesome" attitude.

A few example strings are test:

| String | BERT | Refined to Doma Corpus |
| ------ | ---- | ---------------------- |
| "The closing process should be \<mask\>" | ["stopped", "repeated", "completed", "complete", "done"] | ["thorough", "straightforward", "streamlined", "easy", "transparent"] |
| "The traditional approach to titling is \<mask\>" | ["abandoned", "discontinued", "adopted", "used", "gone"] | ["straightforward", "simple", "outdated", "incorrect", "problematic"] |
| "We want home buying to be \<mask\>" | ["easy", "fun", "good", "free", "done"] | ["affordable", "easy", "simple", "inclusive", "easier"] |
| "With the aid of \<mask\>, we make closing faster" | ["god", "light", "fire", "water", "love"] | ["technology", "software", "data", "computers", "tech"] |

The last one is pretty funny.  In all instances, the model refined to Doma's corpus of marketing materials is better able to fill in sentences such that the text reads like something you would find on their website.
This is not surprising, a good model should be able to adapt to the domain of the language that it's shown.
However, the implications are cool.  Suppose you want to build a chat bot.  The corpus of company-published materials would definitely give the bot a personality that fits the company's vision.