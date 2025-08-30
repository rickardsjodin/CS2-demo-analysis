### Round Swing (and thus Rating 3.0) is Misunderstood – Let's Analyze It More Deeply

##### The Argument for Round Swing

At the start of the round, there's a roughly 50/50 win probability, given equal equipment value. If you single-handedly get the first kill, you shift the probability to your team's advantage – let's say, up to 65%. Therefore, you get credited with that 15% difference. This is intuitive, and everyone is happy.

But let's take a more counter-intuitive example. Say you are in a 1v5 situation and have only a 1% probability of winning the round. Then, you make an awesome 200 IQ play and manage to bring it to a 1v1. Now, we are roughly back to 50/50. This means that so far in the round, you have earned a whopping +49% round swing. But then you die to the last opponent. That means -50%, and you end the round with -1%, exactly the same as if you would have died without any kill. This feels unfair, right? You might have had **400+ damage** output this round, yet you still end with a negative swing rating. Does this really make sense?

Well, look at it this way: from the game's perspective, you lost the round. Winning the match means winning rounds. Factoring out economic impact (let's say the enemies had 16k before the round), it makes virtually no difference to the match outcome whether you died instantly or killed four people before going down. The enemies get a point on the board, and you get none either way. **The objective of the game is not to get kills – they are only a means to an end.** And round swing is an excellent way of highlighting that.

I would even argue that 1v1s are therefore not "overpowered" – they rather show how incredibly important these situations are for the outcome of the game. Once you get to that point, it's all or nothing, which is completely reasonable. It's high-risk, high-reward, in contrast to opening kills, which are lower-reward but also lower-risk. It's also very important that the percentages are fair – I will discuss this more further down.

Therefore, you simply **cannot** look at the scoreboard at the end of the match and say, _"But how can player X have a 1.02 rating while going 25-15 with 120 ADR? This rating system is broken!"_ The rating is completely dependent on the context of the kills, and thereby must be analyzed more closely. And that is a feature, not a bug.

Actually, let's do just that. Yesterday's game between Vitality and Gamer Legion on Train had a very unintuitive scoreboard. The casters even noted it and felt that it was unfair to Kursy. He got a 0.83 rating and a -2.98% swing while top-fragging with 20 kills – 8 more than REZ, who got a 1.01 rating and a +2.22% swing.

![Scoreboard](scoreboard.png)

So let's add a bit of context. I have written some scripts that calculate a simplified version of round swing based on the win percentage table provided in the HLTV article https://www.hltv.org/news/42485/introducing-rating-30.

In my case, REZ ended with a +1.65% swing and Kursy with +0.63%. These are quite far from the HLTV numbers (because of my very simple implementation) but still surprising given their scorelines.

The following graph shows their individual round swing contributions round by round.

![Comparison](comparison.png)

Looking at Kursy, in every round where he made a big positive contribution, he also died shortly after in an important situation. He still ended those rounds with a positive delta, but over the course of the game, it doesn't add up enough. He also had one 3k round, but these were in a 5v3 to 5v0 situation, which only counts for 14.3% in total.

Contrast this with REZ. While having significantly fewer kills overall, his kills had more impact on the ultimate outcome, and he had less impactful deaths. Most notably, a 13% conversion into a round win in round 15 (giving him +87%), and a 1v1 clutch in round 18 (+55.1%). These two rounds gave Gamer Legion two rounds on the board, contributing heavily to the ultimate goal of Counter-Strike: winning rounds.

So the results from Round Swing can be brutal. **But not because the metric itself is brutal – but because the objective of Counter-Strike can be brutal.**

#### Addressing the Underlying Difficulty with Round Swing and Future Potential

Consider the following 1v1 scenario:

`Bomb planted with 15 seconds left, T position unknown, CT no kit and position known.`

This is clearly **not** a 50/50 situation, but a strong advantage for the T. Therefore, if the T wins, he should **not** get a +50% reward, and the CT should **not** get -50%.

So it's obviously important to have accurate probabilities to give a fair impact for each player. Simply looking at the number of players alive is not enough information. Ideally, as discussed in the video, player positions would also be available to make it even more accurate.

The other difficulty is the contribution of both the positives and negatives within the team. For example, the HLTV implementation takes assists and flash assists into account. If you help a teammate get a kill, you get a part of the credit for the extra swing percentage reward. As it is implemented now, rules are set manually. For example, the trading rule: if the enemy that just killed you is killed within 5 seconds, the kill is considered a trade, and you get some of the credit. The obvious weakness here is the arbitrary parameter of 5 seconds.

Today's machine learning models are very mature and relatively easy to get started with (though hard to master, admittedly), and I say this as a Data Scientist myself. I'm honestly a little tempted to train one myself. But I'm going too far now. The main point is that **Round Swing is the ultimate rating when it comes to the objective of Counter-Strike**, provided that you feed it with accurate information. Any current drawback or weakness **is not a weakness of the method itself, but rather a symptom of the inaccuracies in the information it gets fed**.

And if you want to know which player simply fragged the most, you already have stats for that. Everybody's happy!
