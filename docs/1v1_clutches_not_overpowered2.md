#### Why Round Swing 1v1 Clutches Are Not Overpowered

_Note: I'm discussing the **concept** of Round Swing here and are not including limitations in its current implementation (like inaccuracies in probability estimation, inter-round effects etc)._

A common take is that 1v1 clutches are overrated in Round Swing and can mess up the rating in low sample sizes. This is something @ner0 himself has said and it gets repeated over and over again when rating 3.0 is discussed.

The intuitive reaction is understandable: seeing a player get one, sometimes easy, kill and receive a larger rating boost than someone who fought for three kills earlier in the round can feel unbalanced. But this feeling comes from a mix-up between a player's performance and their ultimate impact.

Round swing isn't a "reward" for good plays. Think of it as a simple description of what happened. It answers the question: "Which players' actions had the most impact on the final outcome of the round?" It does not answer, "Who put in the most work in this round?"

To make this distinction crystal clear, consider this scenario: The score is 7-12, and a player on the losing team is left in a 1v5. They put on an incredible display of skill and secure four kills, but narrowly lose the final duel. Despite this heroic performance and outputting 400+ damage, the round was still lost. So even if he had died without a single kill and zero damage output, the final score would still have been 7-13. That player would receive the same low Round Swing score in both cases because the rating isn't measuring their skill; it's describing the fact that they did not ultimately alter the outcome of the round.

This highlights the core idea behind Round Swing: it is intentionally blind to raw performance and focuses solely on impact. The player who wins a 1v1—no matter how messy or simple it was—is the main single reason the round was won instead of lost. Their high rating isn't an overvaluation; it's a simple description of that fact.

We instinctively want a single rating to tell us two stories at once: who was responsible for the win and who played the best. But as the examples show, those are sometimes two different players. People blame these counter-intuitive results on low sample sizes, but this misses the point. The issue isn't the sample size, but that "impact" and "performance" are separate things.

The mismatch between them isn't a flaw; it's an insight. A high-performance, low-impact game tells us a player was brilliant but couldn't convert in the key moments. Conversely, a player with a low K/D but high Round Swing tells us that he made his kills in key moments and did not put his team at a big disadvantage when dying. It provides a richer narrative than if the two numbers were always aligned.

I do agree with the critics on one thing: at large sample sizes, Round Swing is a great skill estimator, since the best-performing players will naturally have the highest impact. At small samples, it is more "noisy."

But this does not mean it's bad at low samples, rather the opposite. That "noise" is the deeper insight.
