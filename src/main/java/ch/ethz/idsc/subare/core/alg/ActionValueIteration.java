// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.red.Max;

/** action value iteration: "policy evaluation is stopped after just one sweep"
 * 
 * Exercise 4.10 on p.91
 * 
 * parallel implementation
 * initial values are set to zeros
 * Jacobi style, i.e. updates take effect only in the next iteration */
public class ActionValueIteration {
  private final StandardModel standardModel;
  private final ActionValueInterface actionValueInterface;
  private final Scalar gamma;
  private DiscreteQsa qsa_new;
  private QsaInterface qsa_old;
  private int iterations = 0;
  private int alternate = 0;

  /** @param standardModel
   * @param gamma discount */
  public ActionValueIteration(StandardModel standardModel, ActionValueInterface actionValueInterface, Scalar gamma) {
    this.standardModel = standardModel;
    this.actionValueInterface = actionValueInterface;
    this.gamma = gamma;
    qsa_new = DiscreteQsa.build(standardModel);
  }

  /** perform iteration until values don't change more than threshold
   * 
   * @param threshold
   * @return */
  public void untilBelow(Scalar threshold) {
    untilBelow(threshold, Integer.MAX_VALUE);
  }

  public void untilBelow(Scalar threshold, int flips) {
    Scalar past = null;
    final long tic = System.nanoTime();
    while (true) {
      step();
      final Scalar delta = qsa_new.distance(qsa_old);
      final long toc = System.nanoTime();
      if (3e9 < toc - tic)
        System.out.println(past + " -> " + delta + " " + alternate);
      if (past != null && Scalars.lessThan(past, delta))
        if (flips < ++alternate) {
          System.out.println("give up at " + past + " -> " + delta);
          break;
        }
      past = delta;
      if (Scalars.lessThan(delta, threshold))
        break;
    }
  }

  /** perform one step of the iteration
   * 
   * @return */
  public void step() {
    qsa_old = qsa_new.copy();
    qsa_new = qsa_new.create(qsa_new.keys().flatten(0) //
        .parallel() //
        .map(pair -> jacobiMax(pair.get(0), pair.get(1))));
    ++iterations;
  }

  // helper function
  private Scalar jacobiMax(Tensor state, Tensor action) {
    Scalar ersa = actionValueInterface.expectedReward(state, action);
    Scalar eqsa = ZeroScalar.get();
    Scalar norm = ZeroScalar.get();
    for (Tensor next : actionValueInterface.transitions(state, action)) {
      Scalar prob = actionValueInterface.transitionProbability(state, action, next);
      Scalar max = standardModel.actions(next).flatten(0) //
          .map(actionN -> qsa_new.value(next, actionN)) //
          .reduce(Max::of).get();
      eqsa = eqsa.add(prob.multiply(max));
      norm = norm.add(prob);
    }
    if (!norm.equals(RealScalar.ONE))
      throw new RuntimeException(); // probabilities have to sum up to 1
    return ersa.add(gamma.multiply(eqsa));
  }

  public DiscreteQsa qsa() {
    return qsa_new;
  }

  public int iterations() {
    return iterations;
  }
}
