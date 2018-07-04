// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.io.IOException;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsaMod;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;

enum TOSm_Gridworld {
  ;
  public static void main(String[] args) throws IOException, Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    FeatureMapper mapper = new ExactFeatureMapper(gridworld);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    TrueOnlineSarsaMod trueOnlineSarsaMod = new TrueOnlineSarsaMod(gridworld, RealScalar.of(0.5), learningRate, mapper);
    Stopwatch stopwatch = Stopwatch.started();
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_tosm.gif"), 250)) {
      for (int episode = 0; episode < 100; ++episode) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        trueOnlineSarsaMod.executeEpisode(RealScalar.of(0.05));
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsa = trueOnlineSarsaMod.getQsa();
        Infoline infoline = Infoline.print(gridworld, episode, ref, qsa);
        gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        if (infoline.isLossfree()) {
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          break;
        }
      }
    }
    System.out.println("Time for TrueOnlineSarsaMod: " + stopwatch.display_seconds() + "s");
  }
}
