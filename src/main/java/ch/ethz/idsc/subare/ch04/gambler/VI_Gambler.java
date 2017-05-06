// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Put;

/** Shangtong Zhang states that using double precision in python
 * "due to tie and precision, can't reproduce the optimal policy in book"
 * 
 * Unlike stated in the book, there is not a unique optimal policy but many
 * using symbolic expressions we can reproduce the policy in book and
 * all other optimal actions
 * 
 * chapter 4, example 3 */
class VI_Gambler {
  public static PolicyInterface getOptimalPolicy(Gambler gambler) {
    Tensor values = new ValueIteration(gambler, RealScalar.ONE).untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobable(gambler, values);
  }

  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, //
        // RealScalar.of(.4)
        RationalScalar.of(40, 100) //
    );
    Tensor values = new ValueIteration(gambler, RealScalar.ONE).untilBelow(RealScalar.of(1e-10));
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobable(gambler, values);
    System.out.println(values);
    Put.of(new File("/home/datahaki/ex403_values"), values);
    greedyPolicy.print(gambler.states());
    // System.out.println(greedyPolicy.policy(RealScalar.of(49), RealScalar.of(1)));
    Tensor greedy = greedyPolicy.flatten(gambler.states);
    Put.of(new File("/home/datahaki/ex403_greedy"), greedy);
  }
}
