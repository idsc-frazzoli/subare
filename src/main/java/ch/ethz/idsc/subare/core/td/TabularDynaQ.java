// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;

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
