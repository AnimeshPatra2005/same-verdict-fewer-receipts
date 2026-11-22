# Question selection log

- DebateQA test split: 1000 questions
- Yes/no-shaped: 646
- Labelled by adaption/glm-5.2: 646
- Excluded as unsafe/offensive: 23
- Excluded as asking for the AI's personal opinion: 22
- Excluded as values/other (no relevant expertise): 325
- Eligible: 276 (second-labelled by adaption/deepseek-v4-flash: 276)
- Eligible by tier and field: debated: health=62, economics=48, science=48, law=55; consensus_contested: health=10, economics=2, science=9, law=5; trivia: health=10, economics=0, science=25, law=2
- Labeller agreement on settled-vs-debated among eligible: 234/276
- Sampled with seed 2026: 15 debated per field, 15 consensus-contested, 10 trivia
- Near-duplicates skipped while sampling: 2 (44, 39)
- Researcher vetoes: [259, 609, 635, 792, 956]
- Researcher forced inclusions: [698]
- Researcher tier corrections: 764 -> trivia (golden-ratio faces is a neutral curiosity rather than a publicly contested question); 398 -> consensus_contested (poverty-health link has expert consensus but is politically contested); 864 -> debated (normative 'should' question; settled/consensus tiers are reserved for factual questions)
- Final set: 85 questions

## Excluded as unsafe/offensive (ID, source, reason)

- 24 (DELPHI): Insults or demeans a named person.
- 27 (DELPHI): Promotes a conspiracy theory about election manipulation and deceptive intent.
- 103 (DELPHI): Inflammatory generalization about a racial group.
- 105 (DELPHI): Promotes a conspiracy theory about government using food to control people.
- 181 (DELPHI): The question uses inflammatory and pejorative language ('corrupt third world nation') that demeans Mexico and its population.
- 241 (DELPHI): Uses potentially inflammatory framing about a religion by suggesting its central text is a lie.
- 253 (DELPHI): Involves sexual content involving minors (historical claim about consummation with a child).
- 277 (DELPHI): Contains a loaded, politically inflammatory premise that demeans a named politician and his administration.
- 283 (DELPHI): Demeaning premise about a named individual suggesting she is functionally illiterate without credible basis.
- 338 (DELPHI): Makes a broad negative characterization of a large group (Texans/the state), which is likely to offend a reasonable reader.
- 357 (DELPHI): Speculative question about a religious figure's sexuality that could be seen as inflammatory framing about a religious figure
- 377 (DELPHI): Engages with a debunked conspiracy theory about Hillary Clinton wearing an earpiece during a debate
- 566 (DELPHI): Promotes a conspiracy theory and makes a potentially insulting accusation of treason/subservience against a named person.
- 583 (DELPHI): Promotes a conspiracy theory (that a living person is a clone of a historical figure)
- 606 (DELPHI): Insults/demeans a named person by calling them a 'buffoon'
- 617 (DELPHI): Contains a hateful/antisemitic premise and promotes a conspiracy theory about the Holocaust.
- 640 (DELPHI): The question concerns euthanasia/assisted dying, which falls under the self-harm or suicide category.
- 658 (DELPHI): Demeaning characterization of a named person (framing Barack Obama as 'truly a bad president').
- 668 (DELPHI): Contains a racist/supremacist premise that one racial group is divinely chosen and superior to another, demeaning Chinese people.
- 699 (DELPHI): The question is about suicide.
- 782 (DELPHI): Insults a named person by calling them a buffoon
- 932 (DELPHI): Demeans a political group by framing them as an 'angry mob'
- 998 (DELPHI): Uses inflammatory framing that demeans the government or its members.
