// code by jph
package ch.ethz.idsc.subare.core.alg;

import java.util.Objects;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ext.Timing;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.sca.N;
import ch.ethz.idsc.tensor.sca.Sign;

/** action value iteration: "policy evaluation is stopped after just one sweep"
 * 
 * Exercise 4.10 on p.91
 * 
 * parallel implementation
 * initial values are set to zeros
 * Jacobi style, i.e. updates take effect only in the next iteration */
public class ActionValueIteration implements DiscreteQsaSupplier {
  /** @param standardModel */
  public static ActionValueIteration of(StandardModel standardModel) {
    return of(standardModel, standardModel);
  }

  public static ActionValueIteration of(StandardModel standardModel, DiscreteQsa qsa_new) {
    return new ActionValueIteration(standardModel, standardModel, qsa_new);
  }

  /** @param discreteModel
   * @param actionValueInterface */
  public static ActionValueIteration of(DiscreteModel discreteModel, ActionValueInterface actionValueInterface) {
    return new ActionValueIteration(discreteModel, actionValueInterface, DiscreteQsa.build(discreteModel));
  }

  // ---
  private final DiscreteModel discreteModel;
  private final ActionValueInterface actionValueInterface;
  private Scalar gamma;
  private DiscreteQsa qsa_new;
  private QsaInterface qsa_old;
  private int iterations = 0;
  private int alternate = 0;

  private ActionValueIteration( //
      DiscreteModel discreteModel, ActionValueInterface actionValueInterface, DiscreteQsa qsa_new) {
    this.discreteModel = discreteModel;
    this.actionValueInterface = actionValueInterface;
    this.gamma = discreteModel.gamma();
    this.qsa_new = qsa_new;
    StaticHelper.assertConsistent(qsa_new.keys(), actionValueInterface);
  }

  /** state-action values are stored in numeric precision */
  public void setMachinePrecision() {
    gamma = N.DOUBLE.of(gamma);
  }

  /** perform iteration until values don't change more than threshold
   * 
   * @param threshold positive */
  public void untilBelow(Scalar threshold) {
    untilBelow(threshold, Integer.MAX_VALUE);
  }

  public void untilBelow(Scalar threshold, int flips) {
    Sign.requirePositive(threshold);
    Scalar past = null;
    Timing timing = Timing.started();
    while (true) {
      step();
      final Scalar delta = DiscreteValueFunctions.distance(qsa_new, (DiscreteQsa) qsa_old);
      if (3E9 < timing.nanoSeconds()) // print info if iteration takes longer than 3 seconds
        System.out.println(past + " -> " + delta + " " + alternate);
      if (Objects.nonNull(past) && Scalars.lessThan(past, delta))
        if (flips < ++alternate) {
          System.out.println("give up at " + past + " -> " + delta);
          break;
        }
      past = delta;
      // TODO JAN consider changing to lessEquals (requires renaming of API functions)
      if (Scalars.lessThan(delta, threshold))
        break;
    }
  }

  /** perform one step of the iteration
   * 
   * @return */
  public void step() {
    qsa_old = qsa_new.copy();
    qsa_new = qsa_new.create(qsa_new.keys().stream() //
        .parallel() //
        .map(pair -> jacobiMax(pair.get(0), pair.get(1))));
    ++iterations;
  }

  // helper function
  private Scalar jacobiMax(Tensor state, Tensor action) {
    Scalar ersa = actionValueInterface.expectedReward(state, action);
    Scalar eqsa = ersa.zero();
    for (Tensor next : actionValueInterface.transitions(state, action)) {
      Scalar prob = actionValueInterface.transitionProbability(state, action, next);
      Scalar max = discreteModel.actions(next).stream() //
          .map(actionN -> qsa_new.value(next, actionN)) //
          .reduce(Max::of).get();
      eqsa = eqsa.add(prob.multiply(max));
    }
    return ersa.add(gamma.multiply(eqsa));
  }

  @Override
  public DiscreteQsa qsa() {
    return qsa_new;
  }

  public int iterations() {
    return iterations;
  }
}
