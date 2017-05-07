// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.TabularTemporalDifference0;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

class TTD0_RandomWalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    PolicyInterface policyInterface = new EquiprobablePolicy(randomWalk);
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        randomWalk, policyInterface, RealScalar.of(.1), RealScalar.ONE, randomWalk);
    Tensor result = ttd0.simulate(123);
    Index statesIndex = Index.build(randomWalk.states);
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND));
    }
  }
}
