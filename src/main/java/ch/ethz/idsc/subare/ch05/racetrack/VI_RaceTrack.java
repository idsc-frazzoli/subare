// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.ExtractPrimitives;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.sca.Round;

class VI_RaceTrack {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track1.png").getPath();
    Tensor image = Import.of(new File(path));
    Racetrack racetrack = new Racetrack(image, 3);
    Tensor actions = racetrack.actions(Tensors.vector(8, 4, 0, 1));
    ValueIteration vi = new ValueIteration(racetrack, RealScalar.of(.5));
    final Tensor values = vi.untilBelow(DecimalScalar.of(2));
    System.out.println("iterations=" + vi.iterations());
    // GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobable(racetrack, values);
    // greedyPolicy.print(racetrack.states());
    Index statesIndex = Index.build(racetrack.states);
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      boolean isStart = racetrack.isStart(state);
      boolean isTerminal = racetrack.isTerminal(state);
      System.out.print(state + " " + racetrack.actions(state).length() + " " + values.get(stateI).map(ROUND) + " ");
      if (isStart)
        System.out.println("start");
      else if (isTerminal)
        System.out.println("final");
      else
        System.out.println();
    }
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(racetrack, values);
    MonteCarloEpisode mce = new MonteCarloEpisode( //
        racetrack, policyInterface, racetrack.statesStart.get(4));
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      int[] index = ExtractPrimitives.toArrayInt(state);
      image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
    }
    Export.of(new File("/home/datahaki/track1_sol.png"), image);
  }
}
