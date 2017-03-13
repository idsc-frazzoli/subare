// code by jph
package ch.ethz.idsc.subare.ch02;

import java.util.function.Function;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Arg;
import ch.ethz.idsc.tensor.alg.Array;

/** The e-greedy agent is described in
 * Section 2.2 "Action-Value Methods" */
public class EGreedyAgent extends Agent {
  final Tensor Na;
  final Tensor Qt;
  final Function<Integer, RealScalar> eps;
  final String string;

  public EGreedyAgent(int n, Function<Integer, RealScalar> eps, String string) {
    Na = Array.zeros(n);
    Qt = Array.zeros(n);
    this.eps = eps;
    this.string = string;
  }

  @Override
  public
  int takeAction() {
    // as described in the algorithm box on p.33
    if (random.nextDouble() < eps.apply(getCount()).getRealDouble())
      return random.nextInt(Qt.numel());
    return Arg.max(Qt); // (2.2)
  }

  @Override
  void protected_feedReward(int a, RealScalar r) {
    // as described in the algorithm box on p.33
    Na.set(NA -> NA.add(RealScalar.of(1)), a);
    Qt.set(QA -> QA.add( //
        r.minus((Scalar) QA).divide(Na.Get(a)) //
    ), a);
  }

  @Override
  public String getDescription() {
    return "a=" + string;
  }
}
