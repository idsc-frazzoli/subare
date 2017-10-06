// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.PriorityQueue;

import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.sca.Sign;

/** Prioritized Sweeping for a deterministic environment
 * 
 * box on p.172 */
public class PrioritizedSweeping implements StepDigest {
  private final Sarsa sarsa;
  private final int n;
  private final Scalar theta;
  private final DeterministicEnvironment deterministicEnvironment = new DeterministicEnvironment();
  private final PriorityQueue<PrioritizedStateAction> priorityQueue = new PriorityQueue<>();
  private final StateOrigins stateOrigins = new StateOrigins();

  /** @param sarsa underlying learning
   * @param n number of replay steps
   * @param theta threshold */
  public PrioritizedSweeping(Sarsa sarsa, int n, Scalar theta) {
    if (Sign.isNegative(theta))
      throw TensorRuntimeException.of(theta);
    this.sarsa = sarsa;
    this.n = n;
    this.theta = theta;
  }
  // public void setPolicy(Policy policy) {
  // sarsa.supplyPolicy(() -> policy);
  // }

  // check priority of learning experience
  private void consider(StepInterface stepInterface) {
    Scalar P = sarsa.priority(stepInterface);
    if (Scalars.lessThan(theta, P))
      priorityQueue.add(new PrioritizedStateAction(P, stepInterface));
  }

  @Override
  public void digest(StepInterface stepInterface) {
    deterministicEnvironment.digest(stepInterface);
    stateOrigins.digest(stepInterface);
    consider(stepInterface);
    // ---
    for (int count = 0; count < n && !priorityQueue.isEmpty(); ++count) {
      PrioritizedStateAction head = priorityQueue.poll();
      final StepInterface model = deterministicEnvironment.get(head.state(), head.action());
      sarsa.digest(model);
      for (StepInterface origin : stateOrigins.values(head.state()))
        consider(origin);
    }
  }
}
