// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.io.Pretty;
import ch.ethz.idsc.tensor.sca.Round;

class VI_RaceTrack {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    Tensor actions = racetrack.actions(Tensors.vector(8, 4, 0, 1));
    System.out.println(Pretty.of(actions));
    ValueIteration vi = new ValueIteration(racetrack, RealScalar.of(.9));
    Tensor values = vi.untilBelow(DecimalScalar.of(.000001));
    System.out.println("iterations=" + vi.iterations());
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobable(racetrack, values);
    // greedyPolicy.print(racetrack.states());
    Index statesIndex = Index.build(racetrack.states);
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      boolean isStart = racetrack.isStart(state);
      boolean isTerminal = racetrack.isTerminal(state);
      System.out.println(state + " " + values.get(stateI).map(ROUND) + " " + isStart + " " + isTerminal);
    }
  }
}
