// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import ch.ethz.idsc.subare.ch02.EGreedyAgent;
import ch.ethz.idsc.subare.ch02.GradientAgent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.ch02.RandomAgent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;

/** chapter 2:
 * Multi-arm Bandits */
class Training {
  @SuppressWarnings("unused")
  static void train(int epochs) {
    final int n = 2;
    RealScalar econst = RationalScalar.of(1, 8);
    Judger judger = new Judger(new Bandits(n), //
        new RandomAgent(n), //
        new GradientAgent(n, RealScalar.of(.1)), //
        new EGreedyAgent(n, i -> econst, econst.toString()), //
        new EGreedyAgent(n, i -> RationalScalar.of(1, i + 1), "1/i"), // yields infty in the first run
        new UCBAgent(n, RealScalar.of(1)), //
        new UCBAgent(n, RealScalar.of(2)), //
        // new GradientAgent(n, 0.25), //
        new OptimistAgent(n, RealScalar.of(1), RealScalar.of(0.1)) //
    );
    // ---
    for (int round = 0; round < epochs; ++round)
      judger.play();
    judger.ranking();
  }

  public static void main(String[] args) {
    train(1000);
  }
}
