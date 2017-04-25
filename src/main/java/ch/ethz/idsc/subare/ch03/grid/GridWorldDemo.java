// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch03.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.BellmanValue;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** produces results on p.65 */
class GridWorldDemo {
  public static void main(String[] args) {
    Function<Scalar, Scalar> round = Round.toMultipleOf(DecimalScalar.of(.1));
    GridWorld gw = new GridWorld();
    Index statesIndex = Index.of(gw.states);
    Tensor result = BellmanValue.iteration( //
        gw, statesIndex, Index.of(gw.actions), DoubleScalar.of(.9), DecimalScalar.of(.0001));
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(round));
    }
  }
}
