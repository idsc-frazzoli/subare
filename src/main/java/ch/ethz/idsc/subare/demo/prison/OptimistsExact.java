// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;

/* package */ class OptimistsExact extends AbstractExact {
  public OptimistsExact(Supplier<Agent> sup1, Supplier<Agent> sup2) {
    super(sup1, sup2);
    // ---
    contribute(new Integer[] { 0 }, new Integer[] { 0 });
    contribute(new Integer[] { 0, 0 }, new Integer[] { 1 });
    contribute(new Integer[] { 0, 1 }, new Integer[] { 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 0, 0 });
    contribute(new Integer[] { 1 }, new Integer[] { 0, 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 1 });
  }
}
