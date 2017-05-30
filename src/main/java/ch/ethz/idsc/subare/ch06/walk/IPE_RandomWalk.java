// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines value function for equiprobable "random" policy
 * 
 * 0 0
 * 1 0.17
 * 2 0.33
 * 3 0.50
 * 4 0.67
 * 5 0.83
 * 6 0 */
class IPE_RandomWalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        randomWalk, new EquiprobablePolicy(randomWalk));
    ipe.until(DecimalScalar.of(.0001));
    ipe.vs().print(ROUND);
  }
}
