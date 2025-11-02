"""Background component with animated network nodes."""

import random
import math
from ..utils.colors import CYAN


class Node:
    """Network node for background animation."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)


class AnimatedBackground:
    """Animated background with gradient and network nodes."""

    def __init__(self, width=900, height=700, node_count=20):
        self.width = width
        self.height = height
        self.nodes = self._create_nodes(node_count)

    def _create_nodes(self, count):
        """Create network nodes."""
        nodes = []
        for _ in range(count):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            nodes.append(Node(x, y))
        return nodes

    def update(self, width, height):
        """Update node positions."""
        for node in self.nodes:
            node.x += node.vx
            node.y += node.vy

            # Bounce off edges
            if node.x < 50 or node.x > width - 50:
                node.vx *= -1
            if node.y < 50 or node.y > height - 50:
                node.vy *= -1

    def draw(self, canvas):
        """Draw background gradient and nodes."""
        w = canvas.winfo_width()
        h = canvas.winfo_height()

        # Gradient background (charcoal to navy)
        for i in range(h):
            ratio = i / h
            r = int(26 + (30 - 26) * ratio)
            g = int(31 + (40 - 31) * ratio)
            b = int(46 + (70 - 46) * ratio)
            canvas.create_line(
                0, i, w, i,
                fill=f"#{r:02x}{g:02x}{b:02x}",
                tags="bg"
            )

        # Draw connections between close nodes
        for i, node1 in enumerate(self.nodes):
            for node2 in self.nodes[i+1:]:
                dist = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
                if dist < 150:
                    alpha = int(100 * (1 - dist / 150))
                    if alpha > 10:
                        canvas.create_line(
                            node1.x, node1.y, node2.x, node2.y,
                            fill=f"#{0:02x}{alpha:02x}{alpha+50:02x}",
                            width=1,
                            tags="bg"
                        )

        # Draw nodes
        for node in self.nodes:
            canvas.create_oval(
                node.x - 3, node.y - 3,
                node.x + 3, node.y + 3,
                fill=CYAN,
                outline="",
                tags="bg"
            )
