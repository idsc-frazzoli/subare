// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import ch.ethz.idsc.subare.ch02.EGreedyAgent;
import ch.ethz.idsc.subare.ch02.GradientAgent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.ch02.RandomAgent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;

/** chapter 2:
 * Multi-arm Bandits */
/* package */ enum Training {
  ;
  static Judger train(int epochs) {
    final int n = 3;
    Scalar econst = RationalScalar.of(1, 12);
    Judger judger = new Judger(new Bandits(n), //
        new RandomAgent(n), //
        new GradientAgent(n, RealScalar.of(.1)), //
        new EGreedyAgent(n, i -> econst, econst.toString()), //
        new EGreedyAgent(n, i -> RationalScalar.of(1, Scalars.intValueExact(i) + 1), "1/i"), new UCBAgent(n, RealScalar.of(1)), //
        new UCBAgent(n, RealScalar.of(1.2)), //
        new UCBAgent(n, RealScalar.of(0.8)), //
        // new GradientAgent(n, 0.25), //
        new OptimistAgent(n, RealScalar.of(1), RealScalar.of(0.1)) //
    );
    // ---
    for (int round = 0; round < epochs; ++round)
      judger.play();
    return judger;
  }

  public static void main(String[] args) {
    Judger judger = train(100);
    judger.ranking();
    // Map<Agent, Tensor> map =
    judger.map();
    // VisualSet visualSet = new VisualSet();
    // for (Entry<Agent, Tensor> entry : map.entrySet()) {
    // // VisualRow visualRow =
    // visualSet.add(Range.of(0, entry.getValue().length()), entry.getValue());
    // // visualRow.setLabel(entry.getKey().getClass().getSimpleName());
    // }
    // JFreeChart jFreeChart = ListPlot.of(visualSet);
    // jFreeChart.setBackgroundPaint(Color.WHITE);
    // ChartUtils.saveChartAsPNG(HomeDirectory.Pictures(Training.class.getSimpleName() + ".png"), jFreeChart, 1280, 720);
  }
}
