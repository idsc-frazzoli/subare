// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;

public class PrioritizedStateAction implements Comparable<PrioritizedStateAction> {
  private final Scalar P;
  private final StepInterface stepInterface;

  public PrioritizedStateAction(Scalar P, StepInterface stepInterface) {
    this.P = P;
    this.stepInterface = stepInterface;
  }

  @Override
  public int compareTo(PrioritizedStateAction prioritizedStateAction) {
    return Scalars.compare(prioritizedStateAction.P, P);
  }

  public Tensor state() {
    return stepInterface.prevState();
  }

  public Tensor action() {
    return stepInterface.action();
  }
}
