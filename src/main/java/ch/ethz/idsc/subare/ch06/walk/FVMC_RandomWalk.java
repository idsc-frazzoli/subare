// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

class FVMC_RandomWalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) {
    RandomWalk gridWorld = new RandomWalk();
    PolicyInterface policyInterface = new EquiprobablePolicy(gridWorld);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gridWorld, policyInterface, RealScalar.ONE, gridWorld);
    Tensor result = fvpe.simulate(123);
    Index statesIndex = Index.build(gridWorld.states);
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND)); // .map(ROUND)
    }
  }
}
