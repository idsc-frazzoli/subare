// code by jph
package ch.ethz.idsc.subare.core.mc;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class MonteCarloExploringStartsTest extends TestCase {
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(monteCarloInterface);
    Policy policy = new EGreedyPolicy(monteCarloInterface, mces.qsa(), RealScalar.of(.1));
    ExploringStarts.batch(monteCarloInterface, policy, mces);
    // DiscreteUtils.print(mces.qsa());
    SimpleTestModels._checkExact(mces.qsa());
  }
}
