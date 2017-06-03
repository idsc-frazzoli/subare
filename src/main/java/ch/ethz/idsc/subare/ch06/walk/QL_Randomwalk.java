// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines state action value function q(s,a).
 * initial policy is irrelevant because each state allows only one action.
 * 
 * {0, 0} 0
 * {1, 0} 0.15
 * {2, 0} 0.32
 * {3, 0} 0.41
 * {4, 0} 0.63
 * {5, 0} 0.84
 * {6, 0} 0 */
class QL_Randomwalk {
  public static void main(String[] args) {
    Randomwalk randomwalk = new Randomwalk();
    DiscreteQsa qsa = DiscreteQsa.build(randomwalk);
    QLearning qLearning = new QLearning( //
        randomwalk, qsa, RealScalar.of(.1)); // TODO ask jz
    // qLearning.simulate(10000); // FIXME
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
