// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.ActionValueStatistics;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

/** finding optimal policy to stay or hit
 * 
 * Figure 5.3 p.108 */
enum AVS_Blackjack {
  ;
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(blackjack);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("blackjack_avs.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 3; // 40
      Tensor epsilon = Subdivide.of(.2, .05, batches);
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
      int episodes = 0;
      ActionValueStatistics avs = new ActionValueStatistics(blackjack);
      for (int index = 0; index < batches; ++index) {
        System.out.println(index + " " + epsilon.Get(index));
        for (int count = 0; count < batches; ++count) {
          episodes += ExploringStarts.batchWithReplay(blackjack, policy, mces, avs);
        }
        ActionValueIteration avi = ActionValueIteration.of(blackjack, avs);
        avi.untilBelow(RealScalar.of(.0001), 3);
        animationWriter.write( //
            Join.of( //
                BlackjackHelper.joinAll(blackjack, mces.qsa()), //
                BlackjackHelper.joinAll(blackjack, avi.qsa())));
        System.out.println(episodes + " " + avs.coverage());
      }
    }
  }
}
