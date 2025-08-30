### Round swing (and thus Rating 3.0) is grievly misunderstood - lets analyze it deeper

In the start of the round, its roughly 50-50 win probability (given equal equipment value).
If you single handedly make the first kill, you shift the probability to your teams advantage - lets say up to 65%. Therefore you get credited that difference of 15%. This is intuitive and everybody is happy.

But lets take a more counter intuitive example. Say you are down 1v5, and only have a 1% probability of winning the round. Then you make some awesome 400 IQ play and manage to make it down to a 1v1. Now we are roughly back to 50-50 again (depending on context). This means that so far in the round, you have earned a whopping 49% round swing. But - you die to the last man. That means -50%, and you end the round with -1%. This feels unfair, right? You might have had 400+ damage output this round and you still end with a minus swing rating? Does this really make sense??

Well, look at it this way: from the games perspective, you still lost the round. Winning the match means winning rounds. If not factoring in economic or perhaps some phycological damage (lets say the enemies had 16k before) it literally does not make a difference to the end match outcome whether you had died directly or killing 4 people before going down. The enemies get a point on the board and you get none either way. **The objective of the game is not kills - its only a means to an end**. And round swing is an excellent way of showing that.

1v1's are therefore not overpowered - they are rather showing how incredibly important these situations are for the outcome of the round. Once you get to that point, its all or nothing on the table and that is completely reasonable. It's high risk - high reward. Opening kills are lower reward but also lower risk. Now, its also very important that the percentages are fair - I will discuss this more later.

Therefore, you simply cannot look at the scoreboard at the end of the match and say "But how can player X have 1.02 in rating while going 25-15? This rating system is broken!!" It is completely dependent on context of the kills, and that is a feature, not a bug.

Actually, lets do just that. Yesterday's game between Vitality and Gamer Legion on train got a very unintuitive scoreboard. The casters even noted it and felt that it was unfair to Kursy. He got a 0.89 rating, -2.98% swing while top-fragging at 20 kills - 8 more than REZ at 1.01 rating and +2.22% swing.

![Scoreboard](scoreboard.png)

So lets add a bit of context to these. I have made some scripts that calculates a simplified version of round swing. I have based it on the win % table provided in the HLTV article https://www.hltv.org/news/42485/introducing-rating-30. I did not get the exact numbers as HLTV (expected since their calculation is more comprehensive), but even in this simplified version we can get some better understanding.

In my case REZ ended with +1.65% swing and Kursy with +0.63%. Quite far from the HLTV numbers but still surprising given their scorelines.

The following graph shows their individual round swing contributions round by round.

![Comparison](comparison.png)

Thi

So the results from Round Swing can be brutal. **But not because the metric itself is brutal - but because the objective of Counter-Strike can be brutal.**
