// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.MathematicaFormat;

class Export {
  static void store(String string, Tensor tensor) {
    Stream<String> stream = MathematicaFormat.of(tensor);
    try {
      Files.write(Paths.get(string), (Iterable<String>) stream::iterator);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    // List<Supplier<Agent>> list = AgentSupplier.getOptimists(.01, .8, 30);
    // List<Supplier<Agent>> list = AgentSupplier.getUCBs(0, 6, 30);
    List<Supplier<Agent>> list = AgentSupplier.getEgreedyC(0.1, .8, 20);
    store("/home/datahaki/egreedyc", AllPairs.performance(list, 20, 500));
    System.out.println("done.");
  }
}
