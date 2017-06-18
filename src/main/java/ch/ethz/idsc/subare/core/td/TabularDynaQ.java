// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DeterministicEnvironment;

/** Tabular Dyna-Q
 * 
 * box on p.172 */
public class TabularDynaQ implements StepDigest {
  private final Sarsa sarsa;
  private final int n;
  private final DeterministicEnvironment deterministicEnvironment = new DeterministicEnvironment();

  /** @param sarsa underlying learning
   * @param n number of replay steps */
  public TabularDynaQ(Sarsa sarsa, int n) {
    this.sarsa = sarsa;
    this.n = n;
  }

  public void setPolicy(Policy policy) {
    sarsa.setPolicy(policy);
  }

  @Override
  public void digest(StepInterface stepInterface) {
    sarsa.digest(stepInterface);
    deterministicEnvironment.digest(stepInterface);
    // replay previously observed steps:
    int min = Math.min(deterministicEnvironment.size(), n);
    for (int count = 0; count < min; ++count)
      sarsa.digest(deterministicEnvironment.getRandomStep());
  }
}
